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
