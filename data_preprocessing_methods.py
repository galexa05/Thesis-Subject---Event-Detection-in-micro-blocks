# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
import re
import codecs
import json
#%matplotlib inline
#pd.set_option('display.max_colwidth', 100)

import hdbscan
import nltk
import numpy as np
import pandas as pd
import random
import re
import spacy
import csv
import sklearn

from mapsplotlib import mapsplot as mplt
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag_sents
from sklearn.cluster import DBSCAN
from sklearn.neighbors import DistanceMetric
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd

import datetime
from time import strptime

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from sklearn.cluster import KMeans
import clustering_models as cl
import timestamp_graphs as tg
import warnings
warnings.filterwarnings('ignore')

def remove_patterns(input_txt,mentions,hashtags,RT):
    m = re.findall(mentions, input_txt)
    h = re.findall(hashtags,input_txt)
    r = re.findall(RT,input_txt)
    u = re.findall(r"http\S+",input_txt)
    mention_list = []
    hashtag_list = []
    url_list = []
    rt1 = 0


    for i in u:
        url_list.append(i)
        #input_txt = re.sub(i,'',input_txt)
    input_txt = re.sub(r"http\S+",'',input_txt)
    for i in m:
        mention_list.append(i)
        input_txt = re.sub(i, '', input_txt)
    for i in h:
        hashtag_list.append(i)
        input_txt = re.sub(i,'',input_txt)
    if r!=0 :
        rt1 = 1
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    #Some json
    x = {"input_text" : input_txt,"user_mentions":mention_list,"hashtags":hashtag_list,"is_RT":rt1,"is_URL":url_list}
    #y = json.dumps(x)
    return x

def list_lower(temp):
    list_lower =[]
    for text in temp:
        list_lower.append(text.lower())
    return list_lower

def remove_punct(text):
    temp = string.punctuation+"…"
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

def tokenization(text):
    text = re.split('\W+', text)
    #text = nltk.word_tokenize(text)
    temp=[]
    for i in text:
        if i!='':
            temp.append(i)
    return temp



def remove_stopwords(text,stopword):
    text = [word for word in text if word not in stopword]
    return text


def stemming_to_sent(words,stemmer):
    stemmed_words = [stemmer.stem(w) for w in words]
    return(" ".join(stemmed_words))


def stemming_to_words(words,stemmer):
    stemmed_words = [stemmer.stem(w) for w in words]
    return stemmed_words

def new_clean_data(i,df_temp):
#     import pdb; pdb.set_trace()
    sent = ""
    for temp in df_temp["Stem_words"].get(i):
        sent+= temp + " "
    for temp in df_temp["hashtag"].get(i):
        sent+= temp + " "
    for temp in df_temp["user_mentions"].get(i):
        sent+= temp+ " "
    if (df_temp['urls'][i]!='None'):
        sent+= df_temp['urls'][i] + " "
    return sent

import pandas as pd
# import geopandas as gpd
# import geopy
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter
import matplotlib.pyplot as plt
# import plotly_express as px

# import reverse_geocoder as rg

def reverse_geocoding(df):
    tweetsContentLocation = df.copy()["clean_data"]
    tweetsContentLocation.shape[0]

    result = [df['coordinates'][tweetsContentLocation.index[i]] for i in range(tweetsContentLocation.shape[0])]
    list_location = []
    for i in range(len(result)):
        row = result[i].replace("\'", "\"")
        row = json.loads(row)
        lat = row['lat']
        lon = row['lon']
        coordinates = (lat,lon)
        list_location.append(coordinates)

    results = rg.search(list_location)
    count = 100000
    for i in range(len(results)):
                    if (i==count):
                        print(i)
                        count+=100000
                    place = tweetsContentLocation.index[i]
                    if results[i]['name']:
                        tweetsContentLocation[place] += results[i]['name'] + ' '
                    if results[i]['admin1']:
                        tweetsContentLocation[place] += results[i]['admin1'] + ' '
                    if results[i]['admin2']:
                        tweetsContentLocation[place] += results[i]['admin2'] + ' '
                    if results[i]['cc']:
                        tweetsContentLocation[place] += results[i]['cc'] + ' '
    return tweetsContentLocation
