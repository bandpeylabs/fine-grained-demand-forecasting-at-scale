// Databricks notebook source
// MAGIC %md This example notebook closely follows the [Databricks documentation](https://docs.azuredatabricks.net/spark/latest/data-sources/azure/azure-datalake.html) for how to set up Azure Data Lake Store as a data source in Databricks.

// COMMAND ----------

// MAGIC %md ### 0 - Setup
// MAGIC
// MAGIC To get set up, do these tasks first: 
// MAGIC
// MAGIC - Get service credentials: Client ID `<aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee>` and Client Credential `<NzQzY2QzYTAtM2I3Zi00NzFmLWI3MGMtMzc4MzRjZmk=>`. Follow the instructions in [Create service principal with portal](https://docs.microsoft.com/en-us/azure/azure-resource-manager/resource-group-create-service-principal-portal). 
// MAGIC - Get directory ID `<ffffffff-gggg-hhhh-iiii-jjjjjjjjjjjj>`: This is also referred to as *tenant ID*. Follow the instructions in [Get tenant ID](https://docs.microsoft.com/en-us/azure/azure-resource-manager/resource-group-create-service-principal-portal#get-tenant-id). 
// MAGIC - If you haven't set up the service app, follow this [tutorial](https://docs.microsoft.com/en-us/azure/azure-databricks/databricks-extract-load-sql-data-warehouse). Set access at the root directory or desired folder level to the service or everyone.

// COMMAND ----------

// MAGIC %md There are two options to read and write Azure Data Lake data from Azure Databricks:
// MAGIC 1. DBFS mount points
// MAGIC 2. Spark configs

// COMMAND ----------

// MAGIC %md ## 1 - DBFS mount points
// MAGIC [DBFS](https://docs.azuredatabricks.net/user-guide/dbfs-databricks-file-system.html) mount points let you mount Azure Data Lake Store for all users in the workspace. Once it is mounted, the data can be accessed directly via a DBFS path from all clusters, without the need for providing credentials every time. The example below shows how to set up a mount point for Azure Data Lake Store.

// COMMAND ----------

val configs = Map(
  "dfs.adls.oauth2.access.token.provider.type" -> "ClientCredential",
  "dfs.adls.oauth2.client.id" -> "<aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee>",
  "dfs.adls.oauth2.credential" -> "<NzQzY2QzYTAtM2I3Zi00NzFmLWI3MGMtMzc4MzRjZmk=>",
  "dfs.adls.oauth2.refresh.url" -> "https://login.microsoftonline.com/<ffffffff-gggg-hhhh-iiii-jjjjjjjjjjjj>/oauth2/token")

dbutils.fs.mount(
  source = "adl://kpadls.azuredatalakestore.net/",
  mountPoint = "/mnt/kp-adls",
  extraConfigs = configs)

// COMMAND ----------

// MAGIC %fs ls /mnt/kp-adls-testing

// COMMAND ----------

// MAGIC %md ##2 - Spark Configs
// MAGIC
// MAGIC With Spark configs, the Azure Data Lake Store settings can be specified per notebook. To keep things simple, the example below includes the credentials in plaintext. However, we strongly discourage you from storing secrets in plaintext. Instead, we recommend storing the credentials as [Databricks Secrets](https://docs.azuredatabricks.net/user-guide/secrets/index.html#secrets-user-guide).
// MAGIC
// MAGIC **Note:** `spark.conf` values are visible only to the DataSet and DataFrames API. If you need access to them from an RDD, refer to the [documentation](https://docs.azuredatabricks.net/spark/latest/data-sources/azure/azure-datalake.html#access-azure-data-lake-store-using-the-rdd-api).

// COMMAND ----------

spark.conf.set("dfs.adls.oauth2.access.token.provider.type", "ClientCredential")
spark.conf.set("dfs.adls.oauth2.client.id", "<aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee>")
spark.conf.set("dfs.adls.oauth2.credential", "<NzQzY2QzYTAtM2I3Zi00NzFmLWI3MGMtMzc4MzRjZmk=>")
spark.conf.set("dfs.adls.oauth2.refresh.url", "https://login.microsoftonline.com/<ffffffff-gggg-hhhh-iiii-jjjjjjjjjjjj>/oauth2/token")

// COMMAND ----------

// MAGIC %fs ls adl://kpadls.azuredatalakestore.net/testing/

// COMMAND ----------

spark.read.parquet("dbfs:/mnt/my-datasets/datasets/iot/events").write.mode("overwrite").parquet("adl://kpadls.azuredatalakestore.net/testing/tmp/kp/v1")

// COMMAND ----------

// MAGIC %fs ls adl://kpadls.azuredatalakestore.net/testing/tmp/kp/v1
