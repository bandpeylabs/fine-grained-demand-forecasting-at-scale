# Databricks notebook source

import re

import logging
logging.getLogger('py4j').setLevel(logging.ERROR)

database = "demand_db"

spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")

spark.sql(f"USE {database}");
