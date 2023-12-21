from confluent_kafka import Consumer, KafkaError
from cassandra.cluster import Cluster

# Kafka consumer configuration
print("start")
consumer_conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'cassandra-consumer',
    'auto.offset.reset': 'earliest'
}

# Kafka topic
kafka_topic = 'live1'

# Cassandra connection
cluster = Cluster(['localhost'])
session = cluster.connect('supplychain')

# Kafka consumer
consumer = Consumer(consumer_conf)
consumer.subscribe([kafka_topic])
print("intiated")
try:
    while True:
        msg = consumer.poll(1.0)

        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                print('Reached end of partition')
            else:
                print('Error while consuming message: {}'.format(msg.error()))
        else:
            # Assuming the message value is a JSON string
            value = msg.value()
            data = eval(value)
            print(data)

            # Insert data into Cassandra
            session.execute("""
                INSERT INTO live_data_full (date, store_nbr, transactions, city, state, type, cluster)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (data['date'], data['store_nbr'], data['transactions'], data['city'], data['state'], data['type'], data['cluster']))

            print(f'Inserted data into Cassandra: {data}')

except KeyboardInterrupt:
    pass  # Allow the program to be terminated with Ctrl+C

finally:
    # Close the Kafka consumer
    consumer.close()

    # Close the Cassandra connection
    cluster.shutdown()

print('Database upload complete')
