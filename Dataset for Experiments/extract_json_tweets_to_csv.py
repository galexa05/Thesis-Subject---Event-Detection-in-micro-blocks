import csv
import codecs
from datetime import datetime
import json
#import requests
import os
import string
import sys
import time

def parse_json_tweet(line):
    tweet = json.loads(line)
    #print(tweet)
    # if tweet['lang'] != 'en':
    #  	#print "non-english tweet:", tweet['lang'], tweet
    #  	return ['', '', '', [], [], []]

    lang = tweet['lang']
    date = tweet['created_at']
    id = tweet['id_str']
    rt = tweet['retweeted']
    
    try:
        urls = tweet['entities']['urls']
        urls = urls[0]['expanded_url']
    except:
        urls = 'None'

    if 'retweeted_status' in tweet:
    	text = tweet['retweeted_status']['full_text']
    else:
    	text = tweet['full_text']

    # print("Hello")
    #print(date,id,text)
    return [text,date,id,rt,lang,urls]

if __name__ == "__main__":
    file_timeordered_json_tweets = codecs.open(sys.argv[1], 'r', 'utf-8')
    fout = codecs.open(sys.argv[2], 'w', 'utf-8')
#     with open('output_relevant_nocord.csv', 'w', newline='') as fout:
    writer = csv.writer(fout)

    temp = 1
    count = 2
    for line in file_timeordered_json_tweets:
        if temp==count:
            print(temp)
            count*=2
        temp+=1
        try:
            [text,created_at,tweet_id,rt,lang,urls] = parse_json_tweet(line)
#             print(urls)
            writer.writerow([text,created_at,tweet_id,rt,lang,urls])
        except:
            pass
    file_timeordered_json_tweets.close()
    fout.close()
