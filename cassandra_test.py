from cassandra.cluster import Cluster

# Specify the Cassandra host and port
cassandra_host = 'localhost'
cassandra_port = 9042  # Replace with your Cassandra port

# Create a Cassandra cluster object
cluster = Cluster([cassandra_host], port=cassandra_port)

try:
    # Connect to the Cassandra cluster
    session = cluster.connect()

    # Print cluster information
    print(f"Connected to Cassandra cluster on host {cassandra_host}, port {cassandra_port}")

except Exception as e:
    print(f"Failed to connect to Cassandra: {str(e)}")

finally:
    # Close the cluster connection
    cluster.shutdown()
