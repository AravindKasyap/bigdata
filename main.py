from os import system
import pyspark
from pyspark.sql import SparkSession
from cassandra.cluster import Cluster
import os
os.environ['HADOOP_HOME'] = 'C://Bigdatalocalsetup//hadoop'


# system.setProperty("hadoop.home.dir", "C://Bigdatalocalsetup//hadoop//bin//")
spark = SparkSession.builder \
    .appName("CassandraImport") \
    .config("spark.cassandra.connection.host", "localhost") \
    .config("spark.cassandra.connection.port", "9042") \
    .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.11:2.5.1") \
    .getOrCreate()
 # .config("spark.jars", "C://Bigdatalocalsetup//spark//jars//pyspark-cassandra-2.4.1.jar")\

keyspace = "supplychain"
table_name = "transactions_table2"

df = spark.read \
    .format("org.apache.spark.sql.cassandra") \
    .options(table=table_name, keyspace=keyspace) \
    .load()

df.show()

spark.stop()
