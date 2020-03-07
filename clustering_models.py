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
import textacy
import csv
import sklearn

from gmplot import gmplot
from mapsplotlib import mapsplot as mplt
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag_sents
from pymprog import *
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


from sklearn.feature_extraction.text import TfidfVectorizer
def getTfIdf(clean_train,windows_corpus,ngram1,ngram2,mindf,maxdf):
    if (mindf==0):
        minimum = 1
    else:
#         minimum = max(int(windows_corpus * 0.0025), 10)
        minimum = 5
    if (maxdf==0):
        maximum = 1.0
    else:
        maximum = 0.8

    tfidf_vectorizer = TfidfVectorizer(
                                       min_df=minimum,
                                       max_df=maximum,
                                       max_features=2000000,
                                       stop_words='english',
                                       use_idf=True,
                                       ngram_range=(ngram1,ngram2)
    )
    # Tf-idf-weighted term-document sparse matrix
    try :
        tfidf_train_data_features = tfidf_vectorizer.fit_transform(clean_train)
    except:
#         print("Hello")
        tfidf_vectorizer = TfidfVectorizer(max_features=2000000,stop_words='english',use_idf=True,ngram_range=(1,3))
        tfidf_train_data_features = tfidf_vectorizer.fit_transform(clean_train)
        minimum = 1
        maximum = 1.0
    finally:
        list_sum = []
        for i in range(0,tfidf_train_data_features.shape[0]):
            list_sum.append(tfidf_train_data_features[i].count_nonzero())

        return {'tfidf_train_data_features':tfidf_train_data_features,'minimum':minimum,'maximum':maximum,'sum':list_sum}

# function to get unique values
def getNumClusters(list1):

    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return len(unique_list)

from sklearn.cluster import AffinityPropagation
def getAffinity(tweets,tfidf):
    clustering = AffinityPropagation().fit(tfidf.toarray())

    clusters = clustering.labels_.tolist()
    temp = clusters.copy()
    temp.remove

    addedCluster = tweets.copy()
    addedCluster['Cluster'] = clusters
    num_clusters = getNumClusters(clusters)

    pd.options.display.max_colwidth = 100

    cls = []
    #num_clusters = 55
    max_num_cluster = 0

    for i in range(0, num_clusters):
        tweetsInCluster = addedCluster[addedCluster['Cluster'] == i]
        if(tweetsInCluster.shape[0] > 0):
            cls.append((tweetsInCluster,tweetsInCluster.shape[0]))

#     cls.sort(key=takeSecond,reverse=True)
    clusters = []
    for i in range(len(cls)):
        clusters.append(cls[i][0])

    return {"clusters":clusters,"labels_pr":clustering.labels_.tolist()}

from sklearn.cluster import KMeans
def getKMeans(tweets,tfidf):
    clustering = KMeans(n_clusters=5,n_jobs=-1).fit(tfidf.toarray())

    clusters = clustering.labels_.tolist()
    temp = clusters.copy()
    temp.remove

    addedCluster = tweets.copy()
    addedCluster['Cluster'] = clusters
    num_clusters = getNumClusters(clusters)

    pd.options.display.max_colwidth = 100

    cls = []
    #num_clusters = 55
    max_num_cluster = 0

    for i in range(0, num_clusters):
        tweetsInCluster = addedCluster[addedCluster['Cluster'] == i]
        if(tweetsInCluster.shape[0] > 0):
            cls.append((tweetsInCluster,tweetsInCluster.shape[0]))

#     cls.sort(key=takeSecond,reverse=True)
    clusters = []
    for i in range(len(cls)):
        clusters.append(cls[i][0])

    return {"clusters":clusters,"labels_pr":clustering.labels_.tolist()}

from sklearn.cluster import AgglomerativeClustering
def getHierarchical(tweets,tfidf):
    clustering = AgglomerativeClustering(n_clusters=5).fit(tfidf.toarray())

    clusters = clustering.labels_.tolist()
    temp = clusters.copy()
    temp.remove

    addedCluster = tweets.copy()
    addedCluster['Cluster'] = clusters
    num_clusters = getNumClusters(clusters)

    pd.options.display.max_colwidth = 100

    cls = []
    #num_clusters = 55
    max_num_cluster = 0

    for i in range(0, num_clusters):
        tweetsInCluster = addedCluster[addedCluster['Cluster'] == i]
        if(tweetsInCluster.shape[0] > 0):
            cls.append((tweetsInCluster,tweetsInCluster.shape[0]))

#     cls.sort(key=takeSecond,reverse=True)
    clusters = []
    for i in range(len(cls)):
        clusters.append(cls[i][0])

    return {"clusters":clusters,"labels_pr":clustering.labels_.tolist()}

def getHDBSCAN(tweets,tfidf):
    clustering = hdbscan.HDBSCAN(min_cluster_size=5,metric='cosine').fit(tfidf)

    clusters = clustering.labels_.tolist()
    temp = clusters.copy()
    temp.remove

    addedCluster = tweets.copy()
    addedCluster['Cluster'] = clusters
    num_clusters = getNumClusters(clusters)

    pd.options.display.max_colwidth = 100

    cls = []
    #num_clusters = 55
    max_num_cluster = 0

    for i in range(0, num_clusters):
        tweetsInCluster = addedCluster[addedCluster['Cluster'] == i]
        if(tweetsInCluster.shape[0] > 0):
            cls.append((tweetsInCluster,tweetsInCluster.shape[0]))

#     cls.sort(key=takeSecond,reverse=True)
    clusters = []
    for i in range(len(cls)):
        clusters.append(cls[i][0])

    #Noise Data
    noise_data = addedCluster[addedCluster['Cluster']==-1]
#     noise_data = addedCluster[addedCluster['Cluster'] == -1]
#     print("Size:",noise_data.shape[0])
#     print(noise_data["tweets"],noise_data["Event"])

    return {"clusters":clusters,"noise_data":noise_data,"labels_pr":clustering.labels_.tolist()}

def getDBSCAN(tweets,tfidf,e):
    clustering = DBSCAN(eps=e, min_samples=5,n_jobs=-1,metric='cosine').fit(tfidf)

    clusters = clustering.labels_.tolist()
    temp = clusters.copy()
    temp.remove

    addedCluster = tweets.copy()
    addedCluster['Cluster'] = clusters
    num_clusters = getNumClusters(clusters)

    pd.options.display.max_colwidth = 100

    cls = []
    #num_clusters = 55
    max_num_cluster = 0

    for i in range(0, num_clusters):
        tweetsInCluster = addedCluster[addedCluster['Cluster'] == i]
        if(tweetsInCluster.shape[0] > 0):
            cls.append((tweetsInCluster,tweetsInCluster.shape[0]))

#     cls.sort(key=takeSecond,reverse=True)
    clusters = []
    for i in range(len(cls)):
        clusters.append(cls[i][0])

    #Noise Data
    noise_data = addedCluster[addedCluster['Cluster']==-1]
#     noise_data = addedCluster[addedCluster['Cluster'] == -1]
#     print("Size:",noise_data.shape[0])
#     print(noise_data["tweets"],noise_data["Event"])

    return {"clusters":clusters,"noise_data":noise_data,"labels_pr":clustering.labels_.tolist()}
