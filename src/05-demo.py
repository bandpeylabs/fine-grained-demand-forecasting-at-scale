# Databricks notebook source
# MAGIC %run "./config/setup"

# COMMAND ----------

trainDF = spark.table("train")

# COMMAND ----------

display(trainDF)

# COMMAND ----------

# MAGIC %md When performing demand forecasting, we are often interested in general trends and seasonality. Let's start our exploration by examining the annual trend in unit sales:
# MAGIC
# MAGIC ### View Yearly Trends

# COMMAND ----------

# MAGIC %sql
# MAGIC  
# MAGIC SELECT
# MAGIC   year(date) as year, 
# MAGIC   sum(sales) as sales
# MAGIC FROM train
# MAGIC GROUP BY year(date)
# MAGIC ORDER BY year;

# COMMAND ----------

# MAGIC %md It's very clear from the data that there is a generally upward trend in total unit sales across the stores. If we had better knowledge of the markets served by these stores, we might wish to identify whether there is a maximum growth capacity we'd expect to approach over the life of our forecast. But without that knowledge and by just quickly eyeballing this dataset, it feels safe to assume that if our goal is to make a forecast a few days, months or even a year out, we might expect continued linear growth over that time span.
# MAGIC
# MAGIC Now let's examine seasonality. If we aggregate the data around the individual months in each year, a distinct yearly seasonal pattern is observed which seems to grow in scale with overall growth in sales:
# MAGIC
# MAGIC ### View Monthly Trends

# COMMAND ----------

# MAGIC %sql
# MAGIC  
# MAGIC SELECT 
# MAGIC   TRUNC(date, 'MM') as month,
# MAGIC   SUM(sales) as sales
# MAGIC FROM train
# MAGIC GROUP BY TRUNC(date, 'MM')
# MAGIC ORDER BY month;

# COMMAND ----------

# MAGIC %md Aggregating the data at a weekday level, a pronounced weekly seasonal pattern is observed with a peak on Sunday (weekday 0), a hard drop on Monday (weekday 1), and then a steady pickup over the week heading back to the Sunday high. This pattern seems to be pretty stable across the five years of observations:
# MAGIC
# MAGIC UPDATE As part of the Spark 3 move to the Proleptic Gregorian calendar, the 'u' option in CAST(DATE_FORMAT(date, 'u') was removed. We are now using 'E to provide us a similar output.
# MAGIC
# MAGIC ### View Weekday Trends
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC  
# MAGIC SELECT
# MAGIC   YEAR(date) as year,
# MAGIC   (
# MAGIC     CASE
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Sun' THEN 0
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Mon' THEN 1
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Tue' THEN 2
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Wed' THEN 3
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Thu' THEN 4
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Fri' THEN 5
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Sat' THEN 6
# MAGIC     END
# MAGIC   ) % 7 as weekday,
# MAGIC   AVG(sales) as sales
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     date,
# MAGIC     SUM(sales) as sales
# MAGIC   FROM train
# MAGIC   GROUP BY date
# MAGIC  ) x
# MAGIC GROUP BY year, weekday
# MAGIC ORDER BY year, weekday;

# COMMAND ----------

# MAGIC %md Now that we are oriented to the basic patterns within our data, let's explore how we might build a forecast.
# MAGIC
# MAGIC ## Step 2: Build a Single Forecast
# MAGIC Before attempting to generate forecasts for individual combinations of stores and items, it might be helpful to build a single forecast for no other reason than to orient ourselves to the use of prophet.
# MAGIC
# MAGIC Our first step is to assemble the historical dataset on which we will train the model:
# MAGIC
# MAGIC ### Retrieve Data for a Single Item-Store Combination

# COMMAND ----------

# query to aggregate data to date (ds) level
sql_statement = '''
  SELECT
    CAST(date as date) as ds,
    sales as y
  FROM train
  WHERE store=1 AND item=1
  ORDER BY ds
  '''
 
# assemble dataset in Pandas dataframe
history_pd = spark.sql(sql_statement).toPandas()
 
# drop any missing records
history_pd = history_pd.dropna()

# COMMAND ----------

# MAGIC %md Now, we will import the prophet library, but because it can be a bit verbose when in use, we will need to fine-tune the logging settings in our environment:
# MAGIC
# MAGIC ### Import Prophet Library

# COMMAND ----------

from prophet import Prophet
import logging
 
# disable informational messages from prophet
logging.getLogger('py4j').setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md Based on our review of the data, it looks like we should set our overall growth pattern to linear and enable the evaluation of weekly and yearly seasonal patterns. We might also wish to set our seasonality mode to multiplicative as the seasonal pattern seems to grow with overall growth in sales:
# MAGIC
# MAGIC ### Train Prophet Model

# COMMAND ----------

# set model parameters
model = Prophet(
  interval_width=0.95,
  growth='linear',
  daily_seasonality=False,
  weekly_seasonality=True,
  yearly_seasonality=True,
  seasonality_mode='multiplicative'
  )
 
# fit the model to historical data
model.fit(history_pd)

# COMMAND ----------

# MAGIC %md Now that we have a trained model, let's use it to build a 90-day forecast:
# MAGIC ### Build Forecast

# COMMAND ----------

# define a dataset including both historical dates & 90-days beyond the last available date
future_pd = model.make_future_dataframe(
  periods=90, 
  freq='d', 
  include_history=True
  )
 
# predict over the dataset
forecast_pd = model.predict(future_pd)
 
display(forecast_pd)

# COMMAND ----------

# MAGIC %md How did our model perform? Here we can see the general and seasonal trends in our model presented as graphs:
# MAGIC
# MAGIC ### Examine Forecast Components

# COMMAND ----------

trends_fig = model.plot_components(forecast_pd)
# display(trends_fig)

# COMMAND ----------

# MAGIC %md And here, we can see how our actual and predicted data line up as well as a forecast for the future, though we will limit our graph to the last year of historical data just to keep it readable:
# MAGIC
# MAGIC ### View Historicals vs. Predictions

# COMMAND ----------

predict_fig = model.plot( forecast_pd, xlabel='date', ylabel='sales')
 
# adjust figure to display dates from last year + the 90 day forecast
xlim = predict_fig.axes[0].get_xlim()
new_xlim = ( xlim[1]-(180.0+365.0), xlim[1]-90.0)
predict_fig.axes[0].set_xlim(new_xlim)
 
# display(predict_fig)

# COMMAND ----------

# MAGIC %md **NOTE** This visualization is a bit busy. Bartosz Mikulski provides [an excellent breakdown](https://www.mikulskibartosz.name/prophet-plot-explained/) of it that is well worth checking out.  In a nutshell, the black dots represent our actuals with the darker blue line representing our predictions and the lighter blue band representing our (95%) uncertainty interval.
# MAGIC
# MAGIC Visual inspection is useful, but a better way to evaluate the forecast is to calculate Mean Absolute Error, Mean Squared Error and Root Mean Squared Error values for the predicted relative to the actual values in our set:
# MAGIC
# MAGIC **UPDATE** A change in pandas functionality requires us to use *pd.to_datetime* to coerce the date string into the right data type.
# MAGIC
# MAGIC ### Calculate Evaluation metrics

# COMMAND ----------

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from datetime import date

# Assuming history_pd and forecast_pd have a 'ds' column with datetime values.
# First, ensure both are DataFrame objects with 'ds' as a DatetimeIndex.
history_pd['ds'] = pd.to_datetime(history_pd['ds'])
forecast_pd['ds'] = pd.to_datetime(forecast_pd['ds'])
history_pd.set_index('ds', inplace=True)
forecast_pd.set_index('ds', inplace=True)

# Now, filter rows where 'ds' is before 2018-01-01 for both DataFrames.
actuals_pd = history_pd[history_pd.index < pd.to_datetime('2018-01-01')]['y']
predicted_pd = forecast_pd[forecast_pd.index < pd.to_datetime('2018-01-01')]['yhat']

# Since there can still be misalignment, align both series by their index.
aligned_actuals, aligned_predicted = actuals_pd.align(predicted_pd, join='inner')

# calculate evaluation metrics using the aligned series
mae = mean_absolute_error(aligned_actuals, aligned_predicted)
mse = mean_squared_error(aligned_actuals, aligned_predicted)
rmse = sqrt(mse)

# print metrics to the screen
print('\n'.join(['MAE: {0}', 'MSE: {1}', 'RMSE: {2}']).format(mae, mse, rmse))

# COMMAND ----------


