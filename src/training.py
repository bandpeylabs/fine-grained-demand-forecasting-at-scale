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


