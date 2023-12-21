from confluent_kafka import Producer
import csv
from cassandra.cluster import Cluster

producer_conf = {
    'bootstrap.servers': 'localhost:9092', 
    'client.id': 'csv-producer'
}
print("initalized")
# # Cassandra connection
# cluster = Cluster(['localhost'])  # Replace with your Cassandra cluster address
# session = cluster.connect('your_keyspace')  # Replace with your Cassandra keyspace name


# kafka_topic = 'supplychaindata'
#kafka_topic='test'
kafka_topic='live1'


#csv_file = 'C://Kasyap//projects//bigdata//data//test//new_transactions.csv'
csv_file = 'C://Kasyap//projects//bigdata//data//consolidated_data.csv'
print("reading data")
# Read data
with open(csv_file, 'r') as file:
    csv_reader = csv.DictReader(file)
    print("sending data")
    for row in csv_reader:
        data = row
        producer = Producer(producer_conf)
        producer.produce(kafka_topic, value=str(data))
        producer.flush()
        print(f'Sent data to Kafka: {data}')
