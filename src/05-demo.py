# Databricks notebook source
# MAGIC %run "./config/setup"

# COMMAND ----------

trainDF = spark.table("train")

# COMMAND ----------

display(trainDF)

# COMMAND ----------

# MAGIC %md When performing demand forecasting, we are often interested in general trends and seasonality. Let's start our exploration by examining the annual trend in unit sales:

# COMMAND ----------


