# Databricks notebook source
# MAGIC %md
# MAGIC #Data Preparation and EDA
# MAGIC  
# MAGIC In this notebook, we'll load our data into Databricks, create our training and test sets, and explore our data.
# MAGIC
# MAGIC ## Details
# MAGIC
# MAGIC - Load data into the workspace using the UI
# MAGIC - Save transformed data as Delta Lake tables
# MAGIC - Use Python to parameterize Spark SQL queries
# MAGIC - Create visualizations with Databricks native plotting tools
# MAGIC - Use widgets to create interactive plots for dashboarding

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Data and Import into Workspace
# MAGIC
# MAGIC For our training dataset, we will make use of 5-years of store-item unit sales data for 50 items across 10 different stores.  This data set is publicly available as part of a past Kaggle competition and can be downloaded [here](https://www.kaggle.com/c/demand-forecasting-kernels-only/data). 
# MAGIC
# MAGIC Once downloaded, we can uzip the *train.csv.zip* file and upload the decompressed CSV to the DBFS:
# MAGIC
# MAGIC ![Upload Train](./docs/images/data-upload.png)
# MAGIC
# MAGIC 1. Click **Catalog** on the sidebar
# MAGIC 2. Click **Browse DBFS** in the top right of the blade that opens
# MAGIC 3. Add `demand_forecasting` to the target directory
# MAGIC 4. Use the **Upload** button to upload your train.csv & test.csv file
# MAGIC 5. Once the upload is complete, you'll see a green check mark next to the path this file uploaded to. Make sure your `rawDataPath` variable in the following cell is updated to match this path.
# MAGIC

# COMMAND ----------

rawTrainDataPath = "/FileStore/demand_forecasting/train.csv"
rawTestDataPath = "/FileStore/demand_forecasting/test.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Setup Script
# MAGIC This create a database

# COMMAND ----------

# MAGIC %run "./config/setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Examine the Data
# MAGIC
# MAGIC Defining the schema and registering a DataFrame from the CSV.

# COMMAND ----------

rawSchema = """
  date DATE,
  store INT,
  item INT,
  sales INT"""

rawTrainDF = (spark.read
  .format("csv") 
  .option("header", True) 
  .schema(rawSchema)
  .load(rawTrainDataPath))

# COMMAND ----------


