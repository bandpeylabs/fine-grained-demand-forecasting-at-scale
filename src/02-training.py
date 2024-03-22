# Databricks notebook source
# MAGIC %md
# MAGIC # Building a Parallel Model
# MAGIC
# MAGIC In this notebook we'll use Facebook Prophet to parallelize the training of time series models for store-item pairs.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Prophet
# MAGIC Use `%pip` to install `prophet` to all nodes in the cluster, but only for this SparkSession.
# MAGIC
# MAGIC This will require a restart of your Python kernel; as such, any variables or imports in the current notebook would be lost. We do this install first to avoid this.

# COMMAND ----------

# MAGIC %pip install prophet

# COMMAND ----------

# MAGIC %md
# MAGIC Run setup for data.

# COMMAND ----------

# MAGIC %run "./config/setup"

# COMMAND ----------

# MAGIC %md
# MAGIC Import the SQL functions for Spark DataFrames.

# COMMAND ----------

import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Train Data into DataFrame

# COMMAND ----------

trainDF = spark.table("train")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generating Store-Item Level Forecasts Sequentially
# MAGIC
# MAGIC With the goal of training a fine-grained model in mind, we will attempt to generate store-item level forecasts as we would on a standard machine without access to distributed data processing capabilities. To get us started, we will create a distinct list of store-item combinations.

# COMMAND ----------

store_items = (
  trainDF
    .select('store', 'item')
    .distinct()
    .collect()
  )

print(f"There are {len(store_items)} unique store-item pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC We will iterate over each store-item combination and for each, extract a pandas DataFrame representing the subset of historical data for that store and item. On that subset-DataFrame, we will train a model and generate a forecast. To facilitate this, we will write a simple function to receive the subset-DataFrame and return the forecast.
# MAGIC
# MAGIC We'll be using Pandas DataFrames within our function.

# COMMAND ----------

import pandas as pd
import logging
logging.getLogger('py4j').setLevel(logging.ERROR)

# Update the import statement from fbprophet to prophet
from prophet import Prophet

# train model & generate forecast from incoming dataset
def get_forecast_sequential(store_item_pd, days_to_forecast):
  
    # retrieve store and item from dataframe
    store = store_item_pd['store'].iloc[0]
    item = store_item_pd['item'].iloc[0]
    
    # configure model
    model = Prophet(
        interval_width=0.95,
        growth='linear',
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='multiplicative'
    )
    
    # fit to dataset
    model.fit( 
        store_item_pd.rename(columns={'date':'ds', 'sales':'y'})[['ds','y']] 
    )
    
    # make forecast
    future_store_item_pd = model.make_future_dataframe(
        periods=days_to_forecast, 
        freq='d',
        include_history=False
    )
    
    # retrieve forecast
    forecast_store_item_pd = model.predict( future_store_item_pd )
    
    # assemble result set 
    forecast_store_item_pd['store'] = store  # assign store field
    forecast_store_item_pd['item'] = item  # assign item field
    
    # return forecast
    return forecast_store_item_pd[
        ['store', 'item', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']
    ]


# COMMAND ----------

# MAGIC %md
# MAGIC Now we can iterate over these store-item combinations, generating a forecast for each.
# MAGIC
# MAGIC A few things to note about this operation:
# MAGIC 1. We are training a single model at a time. With 500 store-item pairs, we will need to wait for the serial execution of 500 models.
# MAGIC 1. By default, Python code will execute on the _driver node only_. This is identical to executing on a single VM.
# MAGIC 1. Some CPU on the driver will be reserved for Databricks tasks, such as managing the metastore. You can track CPU utilization using the

# COMMAND ----------

import pandas as pd  # Ensure pandas is imported

# assemble historical dataset
train_pd = trainDF.toPandas()

# Initialize an empty DataFrame for accumulating results, if not done earlier
store_item_accum_pd = pd.DataFrame()

# for each store-item combination:
for i, store_item in enumerate(store_items):
    print(f"Run {i+1} of {len(store_items)}")
    
    # extract data subset for this store and item
    store_item_train_pd = train_pd[ 
        (train_pd['store'] == store_item['store']) & 
        (train_pd['item'] == store_item['item']) 
    ].dropna()
    
    # fit model on store-item subset and produce forecast
    store_item_forecast_pd = get_forecast_sequential(store_item_train_pd, days_to_forecast=30)
     
    # concatenate forecasts to build a single resultset
    store_item_accum_pd = pd.concat([store_item_accum_pd, store_item_forecast_pd], ignore_index=True)

# At this point, store_item_accum_pd contains all the concatenated forecasts.


# COMMAND ----------

# MAGIC %md
# MAGIC We can quickly see that while calculating predictions for each store-item pair is fairly speedy, even 500 of these models will take roughly 2 minutes. For retailers with many thousands of items over many thousands of stores, this is not a sustainable approach.

# COMMAND ----------

display(store_item_accum_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC While this produces the results we're after, the time it's taking to do so is not supportable. The reliability of timeseries forecasts degrades over time.  We need to generate forecasts far enough out to be actionable but not so far out that they are untrustworthy. For many organizations, this means we need generate weekly, daily or even hourly forecasts based on the latest historical data.  When we consider that a given store may contain 10s or 100s of 1000s of products and then multiply this by the number of stores for which we are generating forecasts, we must have a means to scale this work; processing that volume of data sequentially just isn't a viable solution.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generating Store-Item Level Forecasts in Parallel
# MAGIC
# MAGIC Leveraging Spark and Databricks, we can easily solve this problem.  Instead of iterating over the set of store-item combinations, we will simply group our data by store and item, forcing store-item combinations to be partitioned across the resources in our cluster. To each store-item grouping, we will apply a function, similar to what we did before, to generate a forecast for each combination. The result will be a unified dataset, addressable as a Spark DataFrame.
# MAGIC
# MAGIC To get us started, let's re-write our forecast-generating function so that it may be applied to a Spark DataFrame. What you'll notice is that we are defining this function as a [pandas Grouped Map](https://docs.databricks.com/spark/latest/spark-sql/pandas-function-apis.html#grouped-map) which enables the efficient application of pandas functionality to grouped data in a Spark DataFrame.  But despite the slightly different function signature (which requires us to pre-define the structure of the pandas DataFrame that this function will produce), the internal logic is largely the same as the previous function.

# COMMAND ----------

# Update the import statement from fbprophet to prophet
from prophet import Prophet

# structure of udf result set
result_schema = """
  store INT,
  item INT,
  ds DATE,
  yhat FLOAT,
  yhat_lower FLOAT,
  yhat_upper FLOAT
"""

# get forecast
def get_forecast_spark(keys, grouped_pd):
  
    # drop nan records
    grouped_pd = grouped_pd.dropna()
  
    # identify store and item
    store = keys[0]
    item = keys[1]
    days_to_forecast = keys[2]
  
    # configure model
    model = Prophet(
        interval_width=0.95,
        growth='linear',
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='multiplicative'
    )
  
    # train model
    model.fit(grouped_pd.rename(columns={'date':'ds', 'sales':'y'})[['ds', 'y']])
  
    # make forecast
    future_pd = model.make_future_dataframe(
        periods=days_to_forecast, 
        freq='d', 
        include_history=False
    )
  
    # retrieve forecast
    forecast_pd = model.predict(future_pd)
  
    # assign store and item to group results
    forecast_pd['store'] = store
    forecast_pd['item'] = item
  
    # return results
    return forecast_pd[['store', 'item', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']]


# COMMAND ----------

# MAGIC %md
# MAGIC With our function defined, we can now group our data and apply the function to each group to generate a store-item forecast.
# MAGIC
# MAGIC Passing the literal value `30` as a positional in our `groupBy` statement allows us to pass parameters to the Pandas grouped map function.

# COMMAND ----------

# configure distribution target for groupBy
spark.conf.set("spark.sql.shuffle.partitions", 500 ) 
# the hardcoded value is based on our knowledge of the data
# this is not necessarily a best practice in all scenarios

# generate forecasts
# the days_to_forecast field is used to overcome inability to pass params to pandas udf
store_item_accum_spark = (
  trainDF
    .groupBy('store','item', F.lit(30).alias('days_to_forecast'))
    .applyInPandas(get_forecast_spark, schema=result_schema)
  ).cache()

# action to trigger end-to-end processing (for fair comparison to above)
store_item_accum_spark.count()

# display some results on screen
display(store_item_accum_spark)

# COMMAND ----------

# MAGIC %md
# MAGIC The timings achieved by parallelizing the workload are much better than those achieved with sequential process.  What's nice is that as we add resources to the cluster, we can reduce the overall time required to perform the work.  Here are some sample timings achieved by running the above process on different sized clusters:
# MAGIC
# MAGIC |Cores per VM| Worker Count | Total Worker Cores |  Timing |
# MAGIC |------------|--------------|--------------------|---------|
# MAGIC | 4          | 2            |  8                 | 5.41 m  |
# MAGIC | 4          | 3            | 12                 | 3.03 m  |
# MAGIC | 4          | 4            | 16                 | 2.83 m  |
# MAGIC | 4          | 6            | 24                 | 1.65 m  |
# MAGIC | 4          | 8            | 32                 | 1.36 m  |
# MAGIC
# MAGIC Graphing this data, you can see the general trend as we scale-out the work:

# COMMAND ----------

display(
  spark.createDataFrame( 
    [(8, 5.41), (12, 3.03), (16, 2.83), (24, 1.65), (32, 1.36)], 
    schema=['cores', 'minutes']
    )
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Predictions Using Delta Lake
# MAGIC
# MAGIC In a later notebook, we'll discuss how we might design an idempotent system to store these values in a consumable fashion while maintaining historic predictions. For now, we'll save out our predictions as a Delta Lake table so they can be referenced in later notebooks.

# COMMAND ----------

(store_item_accum_spark.write
  .format("delta")
  .option("path", "/mnt/demand_forecasting/store_item_preds")
  .saveAsTable("store_item_preds")
)
