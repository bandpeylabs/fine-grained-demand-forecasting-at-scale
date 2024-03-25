# Databricks notebook source
# MAGIC %md
# MAGIC # Operationalizing Demand Forecasts
# MAGIC
# MAGIC Now that we have a preferred strategy in mind, it's time to explore how we might operationalize the generation of forecasts in our environment. Requirements for this will vary based on the specific retail-scenario we are engaged in. We need to consider how fast products move, how frequently we have access to new and updated data, how reliable our forecasts remain over time (in the face of new data), and the lead times we require to respond to any insights derived from our forecasts.  But given the scalable patterns we explored in the previous notebooks, processing time for forecast generation is not a limiting factor that should require us to compromise our work.

# COMMAND ----------


