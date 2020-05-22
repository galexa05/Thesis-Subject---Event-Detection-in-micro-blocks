from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from kafka import SimpleProducer, KafkaClient
import json

access_token = "1155762435195854848-OVWea9XpZ0M7x15l6bbogutHJPm7xP"
access_token_secret =  "1RssjQFjvp1tyCAOwCyr9JLhkOyIskp3nv6O2zvqaJaXq"
consumer_key =  "lITrx6VPHNDT2ggtC54o6sJU7"
consumer_secret =  "Qo9jZnz4sBl2D6199K4NdwP63iwe1CShUG32PTcI4QzvNQnZpb"

class StdOutListener(StreamListener):
    # def on_data(self, data):
    #     producer.send_messages("trump", data.encode('utf-8'))
    #     print (data)
    #     return True

    def on_status(self, status):
       if hasattr(status, 'retweeted_status'):
           try:
               tweet = status.retweeted_status.extended_tweet["full_text"]
               id =  status.retweeted_status.id
               screen_name = status.retweeted_status.user.screen_name
               x = {"text":tweet,"id":id,"screen_name":screen_name}
               x = json.dumps(x)
               producer.send_messages("trump",str(x).encode('utf-8'))
               #producer.send_messages("trump", tweet.encode('utf-8'))
               print(x)
           except:
               tweet = status.retweeted_status.text
               id =  status.retweeted_status.id
               screen_name = status.retweeted_status.user.screen_name
               x = {"text":tweet,"id":id,"screen_name":screen_name}
               x = json.dumps(x)
               producer.send_messages("trump",str(x).encode('utf-8'))
               #producer.send_messages("trump", tweet.encode('utf-8'))
               print(x)
       else:
           try:
               tweet = status.extended_tweet["full_text"]
               id =  status.id
               screen_name =  status.user.screen_name
               x = {"text":tweet,"id":id,"screen_name":screen_name}
               x = json.dumps(x)
               producer.send_messages("trump",str(x).encode('utf-8'))
               #producer.send_messages("trump", tweet.encode('utf-8'))
               print(x)
           except:
               tweet = status.text
               id =  status.id
               screen_name =  status.user.screen_name
               x = {"text":tweet,"id":id,"screen_name":screen_name}
               x = json.dumps(x)
               producer.send_messages("trump",str(x).encode('utf-8'))
               #producer.send_messages("trump", tweet.encode('utf-8'))
               print(x)


    def on_error(self, status):
        print (status)

kafka = KafkaClient("localhost:9092")
producer = SimpleProducer(kafka)
l = StdOutListener()
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
stream = Stream(auth, l,tweet_mode='extended')
stream.filter(track="trump")
