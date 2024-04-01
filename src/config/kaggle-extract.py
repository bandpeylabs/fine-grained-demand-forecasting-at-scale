# Databricks notebook source
# MAGIC %md The purpose of this notebook is to download and set up the data we will use for this solution. Before running this notebook, make sure you have entered your own credentials for Kaggle.

# COMMAND ----------

# MAGIC %md
# MAGIC We have manually added our Kaggle credientials in a dedicated Azure Key Vault and created a [secret scope](https://docs.databricks.com/security/secrets/secret-scopes.html)

# COMMAND ----------

dbutils.secrets.listScopes()

# COMMAND ----------

dbutils.secrets.list("emolysis-secret-scope-dev")

# COMMAND ----------

import os
os.environ['kaggle-username'] = dbutils.secrets.get("emolysis-secret-scope-dev", "kaggle-username")
os.environ['kaggle-key'] = dbutils.secrets.get("emolysis-secret-scope-dev", "kaggle-key")

# COMMAND ----------

# MAGIC %md Download the data from Kaggle using the credentials set above:

# COMMAND ----------

# MAGIC %pip install --upgrade kaggle

# COMMAND ----------

# MAGIC %sh 
# MAGIC cd /databricks/driver
# MAGIC export KAGGLE_USERNAME=$kaggle_username
# MAGIC export KAGGLE_KEY=$kaggle_key
# MAGIC kaggle competitions download -c demand-forecasting-kernels-only
# MAGIC unzip -o demand-forecasting-kernels-only.zip

# COMMAND ----------

# MAGIC %md Move the downloaded data to the folder used throughout the accelerator:

# COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/train.csv", "dbfs:/tmp/bandpey/demand_forecasting/train/train.csv")

# COMMAND ----------

import re
from pathlib import Path
# Creating user-specific paths and database names
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
username_sql_compatible = re.sub('\W', '_', useremail.split('@')[0])
tmp_data_path = f"/tmp/fine_grain_forecast/data/{useremail}/"
database_name = f"fine_grain_forecast_{username_sql_compatible}"

# Create user-scoped environment
spark.sql(f"DROP DATABASE IF EXISTS {database_name} CASCADE")
spark.sql(f"CREATE DATABASE {database_name} LOCATION '{tmp_data_path}'")
spark.sql(f"USE {database_name}")
Path(tmp_data_path).mkdir(parents=True, exist_ok=True)
