# Databricks notebook source
# MAGIC %md
# MAGIC ## Comparing Fine-Grain Forecasting to an Allocation Strategy
# MAGIC
# MAGIC Earlier, we did some exploratory work to explain why we should consider store-item level forecasting, arguing that aggregate level evaluation of the data hides some of the variability we might wish to action within our businesses.  Let's now generate an aggregate-level forecast for each product/item and allocate that forecast based on historical sales of those products within the different stores.  This is very typical of how retailers have overcome computational challenges in the past, aggregating sales across stores in a given market or region, and then allocating the results back out under the assumption these stores have similar temporal behaviors.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Prophet
# MAGIC In the last notebook, we used `%pip` to install `prophet` to all nodes in the cluster scoped at the _notebook_ level. As we'll continue to use Prophet in this and the next notebook, we can save a few minutes by installing Prophet as a [cluster library](https://docs.databricks.com/libraries/cluster-libraries.html) using [PyPI](https://docs.databricks.com/libraries/workspace-libraries.html#pypi-libraries). Note that this will change the execution environment for all notebooks attached to the cluster and the library will be re-installed when the cluster is restarted

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
# MAGIC ## Calculate Item-Aggregated Forecasts
# MAGIC
# MAGIC As before, we need to define a function to perform our forecasting work. Here, we'll build our model using daily item sales aggregated across stores.
# MAGIC
# MAGIC
# MAGIC While the function defined in the cell below (and executed in the next cell) will finish executing in around 2 minutes, a slight refactor to this code will allow us to use the `applyInPandas` method to scale out our model training.

# COMMAND ----------

from prophet import Prophet

itemAggSchema = """
  item INT,
  ds DATE,
  yhat FLOAT,
  yhat_lower FLOAT,
  yhat_upper FLOAT
"""

# train model & generate forecast from incoming dataset
def get_forecast_agg_item(keys, groupedDF):
  
  item = keys[0]
  days_to_forecast = keys[1]
  
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
    groupedDF.rename(columns={'date':'ds', 'sales':'y'})[['ds','y']] 
    )
  
  # make forecast
  future_item_pd = model.make_future_dataframe(
    periods=days_to_forecast, 
    freq='d', 
    include_history=False
    )
  
  # retrieve forecast
  forecast_item_pd = model.predict(future_item_pd)
  
  # assemble result set 
  forecast_item_pd['item']=item # assign item field
  
  # return forecast
  return forecast_item_pd[['item', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# COMMAND ----------

# MAGIC %md
# MAGIC Note that we begin our process by summing our item sales across stores. In the last notebook, we set ```spark.conf.set("spark.sql.shuffle.partitions", 500)``` to match the number of store-item pairs in our training set. Make sure you set this configuration with the correct number of unique items before executing your function.

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", 50)

groupedDF = (trainDF
  .groupBy("item", "date")
  .agg(F.sum("sales").alias("sales"))
  )

item_accum_spark = (
  groupedDF
    .groupBy('item', F.lit(30).alias('days_to_forecast'))
    .applyInPandas(get_forecast_agg_item, schema=itemAggSchema)
  ).cache()

item_accum_spark.count()

display(item_accum_spark.head(5))

# COMMAND ----------


