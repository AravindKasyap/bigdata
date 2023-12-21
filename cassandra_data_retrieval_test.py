from cassandra.cluster import Cluster
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Connect to Cassandra
cluster = Cluster(['localhost'])
session = cluster.connect()

# Keyspace and table information
keyspace = "supplychain"
table_name = "live_data_full"

# Use the keyspace
session.set_keyspace(keyspace)

# Execute a CQL query to fetch data from the table
query = f"SELECT * FROM {table_name};"
result = session.execute(query)
for row in result:
    print(row)







cluster.shutdown()


