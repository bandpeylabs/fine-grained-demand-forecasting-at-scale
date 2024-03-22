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

# MAGIC %md
# MAGIC ## Scale Allocation Predictions by Historical Sales Figures
# MAGIC
# MAGIC Now let's calculate the ratio of store-specific sales to total sales for each product.  This will give us the ratios that we need for allocation.

# COMMAND ----------

from pyspark.sql.window import Window

# calculate per-item store sales ratios
item_store_ratios = (
  trainDF
    .groupBy('store','item')
    .agg(F.sum('sales').alias('store_item_sales'))
    .withColumn('item_sales', F.sum('store_item_sales').over(Window.partitionBy('item')))
    .withColumn('ratio', F.expr('store_item_sales/item_sales'))
    .select('item', 'store', 'ratio', 'store_item_sales', 'item_sales')
    .orderBy(['item', 'store'])
  )

display(item_store_ratios)

# COMMAND ----------

# MAGIC %md
# MAGIC Applying these ratios to our cross-store aggregates, we can now produce an allocated forecast for each store-item combination.
# MAGIC
# MAGIC In the cell below, we join our `item_store_ratios` with the predictions we made above. Save out a Delta Lake table `agg_item_preds` with the following schema:
# MAGIC
# MAGIC | field | type |
# MAGIC | --- | --- |
# MAGIC | store | INT |
# MAGIC | item | INT |
# MAGIC | ds | DATE |
# MAGIC | yhat | FLOAT |

# COMMAND ----------

tableName = "agg_item_preds"
tablePath = "/mnt/demand_forecasting/agg_item_preds"

allocatedDF = (item_accum_spark
  .join(item_store_ratios, ["item"], "outer")
  .select("store", "item", "ds", (F.col("yhat") * F.col("ratio")).alias("yhat")))

(allocatedDF.write
  .format("delta")
  .option("path", tablePath)
  .mode("overwrite")
  .saveAsTable(tableName)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate Difference in Model Performance
# MAGIC
# MAGIC The query below will join our `test`, `store_item_preds`, and `agg_item_preds` tables to provide a quick overview of how our predictions vary from each other and the true sales.
# MAGIC
# MAGIC | field | description |
# MAGIC | --- | --- |
# MAGIC | y | true sales |
# MAGIC | yhat | store-item predicted sales |
# MAGIC | yhat_lower | lower bound of 95% confidence interval for store-item predicted sales |
# MAGIC | yhat_upper | upper bound of 95% confidence interval for store-item predicted sales |
# MAGIC | yhat_agg | predicted sales from ratio-allocation on item-aggregated sales |
# MAGIC | pred_diff | the difference between `yhat` and `yhat_agg` |
# MAGIC | miss | difference between true sales and `yhat` |
# MAGIC | agg_miss | difference between true sales and `yhat_agg`

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW pred_overview AS (
# MAGIC   SELECT a.date,
# MAGIC     a.store,
# MAGIC     a.item,
# MAGIC     a.sales AS y,
# MAGIC     b.yhat,
# MAGIC     b.yhat_lower,
# MAGIC     b.yhat_upper,
# MAGIC     c.yhat AS yhat_agg, 
# MAGIC     b.yhat - c.yhat AS pred_diff, 
# MAGIC     a.sales - b.yhat AS miss, 
# MAGIC     a.sales - c.yhat AS agg_miss
# MAGIC   FROM test a
# MAGIC   INNER JOIN store_item_preds b
# MAGIC   ON a.date = b.ds AND a.store=b.store AND a.item=b.item
# MAGIC   INNER JOIN agg_item_preds c
# MAGIC   ON a.date = c.ds AND a.store=c.store AND a.item=c.item
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC But what do these differences really amount to?  How much higher and how much lower are my forecasts when compared to one another? If we just compare our forecasts for the 30-day future period over which we are estimating demand, the differences add up differently by product and by store which leads us to the conclusion that the *goodness* or *badness* of one strategy over the other really depends on the stores and products we are considering, but in some instances there are some sizeable misses that can have implications for out-of-stock and overstock scenarios.
# MAGIC
# MAGIC Let's look at the results for a single item.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM pred_overview
# MAGIC WHERE item = 1
# MAGIC ORDER BY date, store

# COMMAND ----------

# MAGIC %md
# MAGIC Note that if we use a line plot to compare our `miss` and `agg_miss` variables, we can visualize that both models pick up on very similar trends.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate Performance Metrics
# MAGIC
# MAGIC Because we are automating the generation of forecasts, it is important we capture some data with which we can evaluate their reliability. Commonly used metrics include Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).  We will calculate all of these for each store-item forecast. To do this, we will need another function that we will apply to our forecast dataset.

# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error

from math import sqrt

import numpy as np
import pandas as pd

# define evaluation metrics generation function
eval_schema = """
  store INT,
  item INT,
  mse FLOAT,
  rmse FLOAT,
  mae FLOAT,
  model STRING
"""

def evaluate_forecast(keys, forecast_pd):
  
  # get store & item in incoming data set
  store = int(keys[0])
  item = int(keys[1])
  model = keys[2]
  
  # MSE & RMSE
  mse = mean_squared_error( forecast_pd['y'], forecast_pd['yhat'] )
  rmse = sqrt(mse)
  
  # MAE
  mae = mean_absolute_error( forecast_pd['y'], forecast_pd['yhat'] )
  
  # assemble result set
  results = {'store':[store], 'item':[item], 'mse':[mse], 'rmse':[rmse], 'mae':[mae], "model":[model]}
  return pd.DataFrame( data=results )

# COMMAND ----------

# MAGIC %md
# MAGIC We use `applyInPandas` so that we can assess performance for each store-item pair.

# COMMAND ----------

storeItemMetrics = (spark.table("pred_overview")
  .select("store", "item", "y", "yhat")
  .groupBy("store", "item", F.lit("store-item-30day").alias("model"))
  .applyInPandas(evaluate_forecast, schema=eval_schema))

# COMMAND ----------

display(storeItemMetrics)

# COMMAND ----------

aggItemMetrics = (spark.table("pred_overview")
  .select("store", "item", "y", F.col("yhat_agg").alias("yhat"))
  .groupBy("store", "item", F.lit("agg-item").alias("model"))
  .applyInPandas(evaluate_forecast, schema=eval_schema))

# COMMAND ----------

display(aggItemMetrics)

# COMMAND ----------

# MAGIC %md
# MAGIC All of this is not to say that store-item level forecasting is always the right strategy (just like we aren't stating that FB Prophet is the right forecasting tool for every scenario).  What we are saying is that we want the option to explore a variety of forecasting strategies (and techniques).  In the past, computational limitations prohibited this but those days are behind us.
