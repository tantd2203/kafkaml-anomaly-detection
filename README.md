
# kafkaml-anomaly-detection
Project for real-time anomaly detection using kafka and python

It's assumed that zookeeper and kafka are running in the Docker, it follows this process:

- Train an unsupervised machine learning model for anomalies detection
- Save the model to be used in real-time predictions
- Generate fake streaming data and send it to a kafka topic
- Read the topic data with several subscribers to be analyzed by the model
- Predict if the data is an anomaly, if so, send the data to another kafka topic
- Subscribe a slack bot to the last topic to send a message in slack channel if
an anomaly arrives

This could be illustrated as:

![Diagram](./docs/kafka_anomalies.png?style=centerme)

Article explaining how to run this project: [medium](https://towardsdatascience.com/real-time-anomaly-detection-with-apache-kafka-and-python-3a40281c01c9)

* Install Library
```bash
pip install -r requirements.txt
```
* Install Docker Kafka
```bash
docker-compose up
```
* First train the anomaly detection model, run the file:

```bash
model/train.py
```

* Create the required topics

```bash
docker exec kafka kafka-topics --create --bootstrap-server 127.0.0.1:29092 --replication-factor 1 --partitions 1 --topic transactions
docker exec kafka kafka-topics --create --bootstrap-server 127.0.0.1:29092 --replication-factor 1 --partitions 1 --topic anomalies
```

* Check the topics are created

```bash
docker exec kafka kafka-topics --list --bootstrap-server 127.0.0.1:29092
```

* Check file **settings.py** and edit the variables if needed

* Start the producer, run the file

```bash
streaming/producer.py
```

* Start the anomalies detector, run the file

```bash
streaming/anomalies_detector.py
```

* Start sending alerts to Slack, make sure to register the env variable SLACK_API_TOKEN,
then run

```bash
streaming/bot_alerts.py
```
