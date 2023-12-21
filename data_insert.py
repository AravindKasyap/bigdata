import pandas as pd
from cassandra.cluster import Cluster

# CSV file path
csv_file = 'C://Kasyap//projects//bigdata//data//test//new_items.csv'

# Cassandra connection
cluster = Cluster(['localhost'])  # Replace with your Cassandra cluster address
session = cluster.connect('supplychain')  # Replace with your Cassandra keyspace name

# Cassandra table
table_name = 'items'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file)
print("start")
# Iterate through the DataFrame and insert each row into the Cassandra table
for _, row in df.iterrows():
    # Construct your INSERT statement based on your table schema
    # For example, assuming you have 'id', 'name', and 'value' columns in your table
    # insert_statement = f"INSERT INTO {table_name} (store_nbr, city, state,type,cluster) VALUES ({row['store_nbr']}, '{row['city']}', {row['state']}, {row['type']}, {row['cluster']})"
    print(row)
    # Execute the INSERT statement
    session.execute("""
        INSERT INTO items (item_nbr, family, class,perishable)
        VALUES (%s, %s, %s,%s)
        """,
            (str(row['item_nbr']),row['family'],str(row['class']),str(row['perishable'])))
    print(f'Inserted data into Cassandra: {row}')

print("end")
# Close the Cassandra session and cluster connection
session.shutdown()
cluster.shutdown()
