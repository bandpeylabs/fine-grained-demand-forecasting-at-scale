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


