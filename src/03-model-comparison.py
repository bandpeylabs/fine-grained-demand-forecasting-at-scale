# Databricks notebook source
# MAGIC %md
# MAGIC ## Comparing Fine-Grain Forecasting to an Allocation Strategy
# MAGIC
# MAGIC Earlier, we did some exploratory work to explain why we should consider store-item level forecasting, arguing that aggregate level evaluation of the data hides some of the variability we might wish to action within our businesses.  Let's now generate an aggregate-level forecast for each product/item and allocate that forecast based on historical sales of those products within the different stores.  This is very typical of how retailers have overcome computational challenges in the past, aggregating sales across stores in a given market or region, and then allocating the results back out under the assumption these stores have similar temporal behaviors.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Prophet
# MAGIC In the last notebook, we used `%pip` to install `prophet` to all nodes in the cluster scoped at the _notebook_ level. As we'll continue to use Prophet in this and the next notebook, we can save a few minutes by installing Prophet as a [cluster library](https://docs.databricks.com/libraries/cluster-libraries.html) using [PyPI](https://docs.databricks.com/libraries/workspace-libraries.html#pypi-libraries). Note that this will change the execution environment for all notebooks attached to the cluster and the library will be re-installed when the cluster is restarted
