from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Test").getOrCreate()
print("started")
spark.stop()