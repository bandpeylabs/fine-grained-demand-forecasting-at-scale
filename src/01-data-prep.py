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

rawTestDF = (spark.read
  .format("csv") 
  .option("header", True) 
  .schema(rawSchema)
  .load(rawTestDataPath))

# COMMAND ----------

# MAGIC %md
# MAGIC Preview the first 5 lines to make sure that your data loaded correctly.

# COMMAND ----------

display(rawTrainDF.head(5))

# COMMAND ----------

display(rawTestDF)

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ## Create Training and Test Sets
# MAGIC
# MAGIC The test data has all `date` and `store` fields as `null` for easier prediction and evaluation it is better to construct the test set ourselves.
# MAGIC
# MAGIC We'll hold out a month of data for our test set.
# MAGIC
# MAGIC Start by finding the max date for the data.

# COMMAND ----------

import pyspark.sql.functions as F

display(rawTrainDF.select(F.max("date")))

# COMMAND ----------

# MAGIC %md
# MAGIC Create 2 new DataFrames:
# MAGIC - `trainDF`: all but the last month of data
# MAGIC - `testDF`: the date for December 2017

# COMMAND ----------


trainDF = rawTrainDF.filter(F.col("date") < "2017-12-01")
testDF = rawTrainDF.filter(F.col("date") >= "2017-12-01")

print(f"Our training set represents {trainDF.count() / rawTrainDF.count() * 100:.2f}% of the total data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Datasets
# MAGIC
# MAGIC Registering data as a table provides access to queries with SQL and persists transformed data between notebooks.

# COMMAND ----------

tableName_train = "train"
path_train = "mnt/demand_forecasting/train"

tableName_test = "test"
path_test = "mnt/demand_forecasting/test"

# COMMAND ----------

# MAGIC %md
# MAGIC Run the cell below to recursively delete your output directory and drop all tables if re-executing your save logic.

# COMMAND ----------

dbutils.fs.rm("mnt/demand_forecasting/", recurse=True)
spark.sql(f"DROP DATABASE IF EXISTS {database} CASCADE")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The function will create an unmanaged table using Delta Lake and accept three arguments:
# MAGIC - a DataFrame
# MAGIC - a path
# MAGIC - a table name

# COMMAND ----------

from pyspark.sql import DataFrame

def saveDeltaTable(df:DataFrame, path:str, tableName:str):
  (df.write
    .format("delta")
    .option("path", path)
    .saveAsTable(tableName)
  )

# COMMAND ----------

saveDeltaTable(trainDF, path_train, tableName_train)
saveDeltaTable(testDF, path_test, tableName_test)

# COMMAND ----------

dbutils.fs.ls(path_train)

# COMMAND ----------


