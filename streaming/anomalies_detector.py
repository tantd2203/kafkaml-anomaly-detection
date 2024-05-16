import json
import os
from joblib import load
import logging
from multiprocessing import Process

import numpy as np

from streaming.utils import create_producer, create_consumer
from settings import TRANSACTIONS_TOPIC, TRANSACTIONS_CONSUMER_GROUP, ANOMALIES_TOPIC, NUM_PARTITIONS

# Path to the trained Isolation Forest model
model_path = os.path.abspath('../model/isolation_forest.joblib')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def detect():
    # Create a Kafka consumer for the transactions topic
    consumer = create_consumer(topic=TRANSACTIONS_TOPIC, group_id=TRANSACTIONS_CONSUMER_GROUP)
    # Create a Kafka producer to send messages to the anomalies topic
    producer = create_producer()

    # Load the pre-trained Isolation Forest model
    clf = load(model_path)

    logging.info("Consumer started and model loaded")

    while True:
        # Poll for new messages from Kafka
        message = consumer.poll(timeout=1.0)
        if message is None:
            continue

        if message.error():
            logging.error("Consumer error: {}".format(message.error()))
            continue

        try:
            # Parse the JSON message
            record = json.loads(message.value().decode('utf-8'))
            data = np.array(record["data"]).reshape(1, -1)
            print(data)

            # Predict if the transaction is normal or an anomaly
            prediction = clf.predict(data)
            if prediction[0] ==1:
                print("Normal")

            if prediction[0] == -1:  # Anomaly detected
                print("Anomaly")
                score = clf.score_samples(data)
                record["score"] = np.round(score, 3).tolist()

                # Convert the record back to JSON
                record = json.dumps(record).encode("utf-8")

                # Send the anomaly record to the anomalies topic
                producer.produce(topic=ANOMALIES_TOPIC, value=record)
                producer.flush()
                print(record)
                print('Alert sent!')

            # Commit the message if processing is successful
            consumer.commit(message)

        except Exception as e:
            logging.error(f"Error processing message: {e}")

    consumer.close()


if __name__ == '__main__':
    # Start a consumer process for each partition
    processes = []
    for _ in range(NUM_PARTITIONS):
        p = Process(target=detect)
        p.start()
        processes.append(p)

    # Ensure all processes are properly joined
    for p in processes:
        p.join()
