# Databricks notebook source
# MAGIC %run "./config/setup"

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
