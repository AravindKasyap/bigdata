from cassandra.cluster import Cluster

cluster = Cluster(['localhost'])

session = cluster.connect('supplychain')

query = "SELECT * FROM transactions_table2"

result = session.execute(query)

for row in result:
    id_value = row.id
    date_value = row.date
    store_nbr_value = row.store_nbr
    item_nbr_value = row.item_nbr
    onpromotion_value = row.onpromotion

    print(f"ID: {id_value}, Date: {date_value}, Store Nbr: {store_nbr_value}, Item Nbr: {item_nbr_value}, On Promotion: {onpromotion_value}")

session.shutdown()
cluster.shutdown()
