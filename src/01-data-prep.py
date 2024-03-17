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

# MAGIC %md
# MAGIC ## Examine the Data
# MAGIC We'll use our training data during EDA; as such, our trend analyses will be missing the month of December 2017.
# MAGIC
# MAGIC Note that the results of SQL queries in Databricks notebooks are equivalent to the results of a `display()` function on a DataFrame. We'll use to visualize our trends.
# MAGIC
# MAGIC When performing demand forecasting, we are often interested in general trends and seasonality. Let's start our exploration by examing the annual trend in unit sales:

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

# MAGIC %md
# MAGIC It's very clear from the data that there is a generally upward trend in total unit sales across the stores. If we had better knowledge of the markets served by these stores, we might wish to identify whether there is a maximum growth capacity we'd expect to approach over the life of our forecast.  But without that knowledge and by just quickly eyeballing this dataset, it feels safe to assume that if our goal is to make a forecast a few days, weeks or months from our last observation, we might expect continued linear growth over that time span.
# MAGIC
# MAGIC Now let's examine seasonality.  If we aggregate the data around the individual months in each year, a distinct yearly seasonal pattern is observed which seems to grow in scale with overall growth in sales:

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

# MAGIC %md
# MAGIC Aggregating the data at a weekday level, a pronounced weekly seasonal pattern is observed with a peak on Sunday (weekday 1), a hard drop on Monday (weekday 2) and then a steady pickup over the week heading back to the Sunday high. This pattern seems to be pretty stable across the five years of observations, increasing with scale given the overall annual trend in sales growth:

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC   YEAR(date) as year,
# MAGIC   DAYOFWEEK(date) % 7 as weekday,
# MAGIC   AVG(sales) as sales
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     date,
# MAGIC     SUM(sales) as sales
# MAGIC   FROM train
# MAGIC   GROUP BY date
# MAGIC  ) x
# MAGIC GROUP BY year, DAYOFWEEK(date)
# MAGIC ORDER BY year, weekday;

# COMMAND ----------

# MAGIC %md
# MAGIC But as mentioned at the top of this section, there are 10 stores represented in this dataset, and while these stores seem to behave in similar manners, there is still some variation between them as can be seen with a close examination of average monthly sales between stores:

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC   MONTH(month) as month,
# MAGIC   store,
# MAGIC   AVG(sales) as sales
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     TRUNC(date, 'MM') as month,
# MAGIC     store,
# MAGIC     SUM(sales) as sales
# MAGIC   FROM train
# MAGIC   GROUP BY TRUNC(date, 'MM'), store
# MAGIC   ) x
# MAGIC GROUP BY MONTH(month), store
# MAGIC ORDER BY month, store;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameterize a SQL Query
# MAGIC
# MAGIC Below, a Python method that returns monthly sales for a single item, grouped by store. The function accepts a single argument for item `sku`.

# COMMAND ----------

def monthlyItemSales(sku:int):
  display(spark.sql(f"""
    SELECT
      MONTH(month) as month,
      store,
      AVG(sales) as sales
    FROM (
      SELECT 
        TRUNC(date, 'MM') as month,
        store,
        SUM(sales) as sales
      FROM train
      WHERE item={sku}
      GROUP BY TRUNC(date, 'MM'), store
      ) x
    GROUP BY MONTH(month), store
    ORDER BY month, store;
   """))

# COMMAND ----------

# MAGIC %md
# MAGIC The cell below defined a dropdown widget that includes all 50 of our item SKUs.

# COMMAND ----------

dbutils.widgets.dropdown("sku", "1", [str(x) for x in range(1,51)])

# COMMAND ----------

# MAGIC %md
# MAGIC Use the function to visually explore trends of several items. Make a line chart with the following specifications:
# MAGIC - keys: `month`
# MAGIC - series groupings: `store`
# MAGIC - values: `sales`

# COMMAND ----------

monthlyItemSales(dbutils.widgets.get("sku"))

# COMMAND ----------

# MAGIC %md
# MAGIC This data exploration reveals the fundamental issue we wish to tackle in this demonstration.  When we examine our data at an aggregate level, i.e. summing sales across stores or individual products, the aggregation hides what could be some useful variability in our data.  By analyzing our data at the store-item level, we hope to be able to construct forecasts capable of picking up on these subtle differences in behaviours.
# MAGIC
