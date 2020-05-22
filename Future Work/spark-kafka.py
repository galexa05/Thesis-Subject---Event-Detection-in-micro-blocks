#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import os
# os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars org.apache.spark:spark-streaming-kafka-0-8_2.11:2.0.2spark-streaming-kafka-0-8-assembly_2.11-2.4.4.jar pyspark-shell'


# In[2]:

import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.0.2 pyspark-shell'


# In[3]:


#    Spark
from pyspark import SparkContext
#    Spark Streaming
from pyspark.streaming import StreamingContext
#    Kafka
from pyspark.streaming.kafka import KafkaUtils
#    json parsing
import json


# In[4]:


sc = SparkContext(appName="PythonStreamingDirectKafkaWordCount")
sc.setLogLevel("WARN")

# In[5]:


ssc = StreamingContext(sc, 10)


# In[6]:


topic = "trump"
brokers = "localhost:9092"

kafkaStream = KafkaUtils.createDirectStream(ssc, [topic],{"metadata.broker.list": brokers})

# In[7]:


parsed = kafkaStream.map(lambda v: json.loads(v[1]))

batch = parsed.count().map(lambda x:'Tweets in this batch: %s' % x)
batch.pprint()

def getTweet(tweet):
    return tweet['text']

temp = parsed.map(lambda tweet: getTweet(tweet))
counts = temp.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)
counts.pprint()

authors_dstream = parsed.map(lambda tweet: tweet['screen_name'])
author_counts = authors_dstream.countByValue()
author_counts.pprint()


#author_counts_sorted_dstream = author_counts.transform((lambda foo:foo.sortBy(lambda x:( -x[1]))))
#author_counts_sorted_dstream.pprint()

# parsed.\
#     flatMap(lambda tweet:tweet['text'].split(" "))\
#     .countByValue()\
#     .transform\
#       (lambda rdd:rdd.sortBy(lambda x:-x[1]))\
#     .pprint()

ssc.start()
ssc.awaitTermination()


# In[ ]:
