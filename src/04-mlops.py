# Databricks notebook source
# MAGIC %md
# MAGIC # Operationalizing Demand Forecasts
# MAGIC
# MAGIC Now that we have a preferred strategy in mind, it's time to explore how we might operationalize the generation of forecasts in our environment. Requirements for this will vary based on the specific retail-scenario we are engaged in. We need to consider how fast products move, how frequently we have access to new and updated data, how reliable our forecasts remain over time (in the face of new data), and the lead times we require to respond to any insights derived from our forecasts.  But given the scalable patterns we explored in the previous notebooks, processing time for forecast generation is not a limiting factor that should require us to compromise our work.

# COMMAND ----------

# MAGIC %run "./config/setup"

# COMMAND ----------

# MAGIC %md
# MAGIC Import the SQL functions for Spark DataFrames.

# COMMAND ----------

import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC We'll be making predictions on store-item pairs, of which we have 500 in our dataset. We'll set our shuffle partitions accordingly.

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", 500 )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting Up the Simulation
# MAGIC
# MAGIC For our demonstration, let's say that we've decided to generate new forecasts every day for each of our store-item combinations.  In this scenario, we might expect that we receive new sales data on a daily basis and that we use that new data in combination with our other historical data to generate forecasts at the start of the next day.
# MAGIC
# MAGIC The dataset with which we are working has data from **January 1, 2013 through December 31, 2017**.  To simulate our movement through time, we will start our demonstration on **December 1, 2017**, generating forecasts using data from the start of 2013 up to **November 30, 2017**, the day prior, and then repeat this process for each day for a week. (Note that we could continue to make predictions through the end of December, but without scaling up our cluster this simulation will be quite lengthy.)
# MAGIC
# MAGIC To assist with this movement through time, let's generate a date range object, the members of which we can iterate over.

# COMMAND ----------

from datetime import timedelta, date

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)
        
print( 
  list(daterange( date(2017,12,1), date(2017,12,8)))
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Now we need a function which we can apply to our Spark DataFrame in order to generate our forecast.  We'll reuse the function we employed in notebook 2 for parallel model training and prediction.

# COMMAND ----------

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
  model.fit( grouped_pd.rename(columns={'date':'ds', 'sales':'y'})[['ds','y']]  )
  
  # make forecast
  future_pd = model.make_future_dataframe(
    periods=days_to_forecast, 
    freq='d', 
    include_history=False
    )
  
  # retrieve forecast
  forecast_pd = model.predict( future_pd )
  
  # assign store and item to group results
  forecast_pd['store']=store
  forecast_pd['item']=item
  
  # return results
  return forecast_pd[['store', 'item', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulate Data Arrival
# MAGIC
# MAGIC One of the benefits of the ACID properties of Delta Lake is that newly arriving data will automatically be included in subsequent reads of a table. We'll take advantage of this behavior to load a new day of data for each model that we train.
# MAGIC
# MAGIC We'll start by loading our `train` and `test` tables. We'll create a new table `prod` for use in our simulation.

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS prod")
dbutils.fs.rm("/mnt/demand_forecasting/prod", recurse=True)

trainDF = spark.table("train")
testDF = spark.table("test")
(trainDF.write
  .format("delta")
  .mode("overwrite")
  .option("path", "/mnt/demand_forecasting/prod")
  .saveAsTable("prod"))
prodDF = spark.table("prod")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Running the Simulation
# MAGIC
# MAGIC In this simulation, to maximize our accuracy, we will only predict sales for the next day. Prior to training our new model, we append the day's sales figures to our training data. This simulation represents the pattern that might be used for daily forecasting: after new data arrives, we run a job to forecast likely sales for the next day for each store, which then guides local distribution of goods from regional distribution centers.
# MAGIC
# MAGIC In this simulation, we'll combine [Delta Lake table versioning](https://docs.databricks.com/delta/delta-batch.html#query-an-older-snapshot-of-a-table-time-travel) with [MLflow experiment tracking](https://docs.databricks.com/applications/mlflow/tracking.html#notebook-experiments). This allows us to recreate the environment in which our model was trained for reproducible data science.

# COMMAND ----------

# MAGIC %md
# MAGIC The following cell is provided to reset the predictions.

# COMMAND ----------

# spark.sql("DROP TABLE IF EXISTS forecasts")
# dbutils.fs.rm(userhome + '/demand/forecasts', True)

# COMMAND ----------


