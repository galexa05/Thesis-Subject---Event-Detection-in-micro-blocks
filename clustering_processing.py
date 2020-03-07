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
from sklearn.preprocessing import StandardScaler
import statistics

def getClusterLabel(temp_cluster):
    count_df = temp_cluster.groupby(['Cluster']).count()
    label = count_df.sort_values(by=['tweets'],ascending=False).index[0]
    return label

def create_clusters_time_window(df,ngram,epsilon,mindf,maxdf,coordinates,method_cl,list_date_minutes):
    temporal_clusters = None
    saved_clusters = None
    df['Clustered'] = False
    df['Cluster'] = -1
    count_labels = 0
    timestamp = 0
    count_timestamp = 1000
    list_cluster_timer = []
    list_cluster_cycles = []
#     for k in range(0,len(list_date_minutes)-10):
    for k in range(20):
        timestamp+=1
        if timestamp == count_timestamp:
            print('Timestamp:',timestamp)
            count_timestamp+=1000
        for i in range(len(list_cluster_timer)):
            list_cluster_timer[i]+=1
            list_cluster_cycles[i] += 1
        df1 = df.loc[df['Datetime']==list_date_minutes[k]]
        df2 = df.loc[df['Datetime']==list_date_minutes[k+1]]
        df3 = df.loc[df['Datetime']==list_date_minutes[k+2]]
        df4 = df.loc[df['Datetime']==list_date_minutes[k+3]]
        df5 = df.loc[df['Datetime']==list_date_minutes[k+4]]
        df6 = df.loc[df['Datetime']==list_date_minutes[k+5]]
        df7 = df.loc[df['Datetime']==list_date_minutes[k+6]]
        df8 = df.loc[df['Datetime']==list_date_minutes[k+7]]
        df9 = df.loc[df['Datetime']==list_date_minutes[k+8]]
        df10 = df.loc[df['Datetime']==list_date_minutes[k+9]]
        list_of_dataframes = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,temporal_clusters]
        dfchange = pd.concat(list_of_dataframes)
        dfchange = dfchange.sort_values(by=['Datetime'])

        tweets = dfchange.sort_values(by=['time'])
        if coordinates == False:
            tweetsContent = tweets.copy()["clean_data"]
        else:
            tweetsContent = tweets.copy()['clean_data_coordinates']

        tfidf_object = cl.getTfIdf(tweetsContent,dfchange.shape[0],ngram,3,mindf,maxdf)
        tfidf = tfidf_object['tfidf_train_data_features']
        minimum = tfidf_object['minimum']
        maximum = tfidf_object['maximum']
        list_sum = tfidf_object['sum']
        tweets['sum'] = list_sum
        noise_tweets = tweets.loc[tweets['sum']==0]

#         for index,row in noise_tweets.iterrows():
#             list_y_fscore.append(row["Event"])
#             list_pr_fscore.append(-1)

        train_tweets = tweets.loc[tweets['sum']!=0]
        if coordinates == False:
            trainContent = train_tweets.copy()["clean_data"]
        else:
            trainContent = train_tweets.copy()['clean_data_coordinates']

        headline_vectorizer = CountVectorizer(binary=True, min_df=1,ngram_range=(1,1))
        tfidf = headline_vectorizer.fit_transform(trainContent)

        if (method_cl == 'dbscan'):
            x_object = cl.getDBSCAN(train_tweets,tfidf,epsilon)
        elif (method_cl == 'hdbscan'):
            x_object = cl.getHDBSCAN(train_tweets,tfidf)
        elif (method_cl == 'hierarchical'):
            x_object = cl.getHierarchical(train_tweets,tfidf)
        elif (method_cl == 'kmeans'):
            x_object = cl.getKMeans(train_tweets,tfidf)
        else:
            x_object = cl.getAffinity(train_tweets,tfidf)

        candidate_cluster = x_object["clusters"]
        if (method_cl=='dbscan' or method_cl=='hdbscan'):
                candidate_noise = x_object["noise_data"]

        new_candidates = []
        for temp_cluster in candidate_cluster:
            new_candidates.append(temp_cluster)

        for temp in new_candidates:
            temp_cluster = temp.loc[temp['Clustered']==True]
            if temp_cluster.shape[0]==0:
                df = df.drop(temp.index.tolist())
                temp['Clustered'] = True
                temp['Cluster'] = count_labels
                list_cluster_timer.append(0)
                list_cluster_cycles.append(0)

                count_labels+=1
                temporal_clusters = pd.concat([temporal_clusters,temp])
            else:
                label = getClusterLabel(temp_cluster)
                label_cluster = temporal_clusters.loc[temporal_clusters['Cluster']==label]
                if temp.shape[0] != label_cluster.shape[0]:
#                     new_items = temp.loc[temp['Clustered']==False]
                    new_items = temp.loc[temp['Cluster']!=label]
                    df = df.drop(new_items.index.tolist())
                    temporal_clusters = temporal_clusters.drop(new_items.index.tolist())
                    new_items['Clustered'] = True
                    new_items['Cluster'] = label
                    temporal_clusters = pd.concat([temporal_clusters,new_items])
                    list_cluster_timer[label] = 0

        for i in range(len(list_cluster_timer)):
            if(list_cluster_timer[i]==3 or list_cluster_cycles[i]==10):
#                 import pdb; pdb.set_trace()
                temp = temporal_clusters.loc[temporal_clusters['Cluster']==i]
                if temp.shape[0]!=0:
                    temp['Timestamp'] = timestamp
                    temp_df = temp[['tweets','coordinates','Timestamp','Cluster','Event']]
                    saved_clusters = pd.concat([saved_clusters,temp_df])
                    # print('Timestamp:',list_date_minutes[k],'-',list_date_minutes[k+9])
                    # print('Size: ',temp.shape[0])
                    # for index,row in temp.iterrows():
                    #     print(row['Event'],' ',end='')
                    # print()
                    # print()
                temporal_clusters = temporal_clusters.drop(temp.index.tolist())
                list_cluster_timer[i]=-1
                list_cluster_cycles[i] = -1
    return saved_clusters

def getLabel(list_event):
    temp = list_event.value_counts()
    return temp.keys()[0]

def getPurity(current_cluster,pr_event) :
    cluster_length = current_cluster.shape[0]
    purity_size = current_cluster.loc[current_cluster['Event']==pr_event].shape[0]
    return purity_size/cluster_length


def create_clusters_hourly(df,ngram,epsilon,mindf,maxdf,coordinates,method_cl,list_date_hour):
    list_y_fscore = []
    list_pr_fscore = []
    adjusted_mutual_info = []
    adjusted_rand_score = []
    completeness_score = []
    fowlkes_mallows_score = []
    homogeneity_score = []
    mutual_info_score = []
    normalized_mutual_info_score = []
    v_measure_score = []
    silhouette = []
    calinski_harabasz = []
    davies_bouldin = []
    contingency = []
    count = 0
    count_labels = 0
    return_df = None
    specs_cluster_df = pd.DataFrame(columns=['Cluster_ID', 'Cluster_Length', 'Predicted_Event','Purity','Timestamp'])
    for current_date in list_date_hour:
        try:
            list_y = []
            list_pr = []

            dfchange = df.loc[df['DateHour']==current_date]
            tweets = dfchange.sort_values(by=['time'])
            if coordinates == False:
                tweetsContent = tweets.copy()["clean_data"]
            else:
                tweetsContent = tweets.copy()['clean_data_coordinates']
            tfidf_object = cl.getTfIdf(tweetsContent,dfchange.shape[0],ngram,3,mindf,maxdf)
            tfidf = tfidf_object['tfidf_train_data_features']
            minimum = tfidf_object['minimum']
            maximum = tfidf_object['maximum']
            list_sum = tfidf_object['sum']
            tweets['sum'] = list_sum
            noise_tweets = tweets.loc[tweets['sum']==0]

            count+=1
            for index,row in noise_tweets.iterrows():
                list_y_fscore.append(row["Event"])
                list_pr_fscore.append(-1)

            train_tweets = tweets.loc[tweets['sum']!=0]
            if coordinates == False:
                trainContent = train_tweets.copy()["clean_data"]
            else:
                trainContent = train_tweets.copy()['clean_data_coordinates']

            headline_vectorizer = CountVectorizer(binary=True, min_df=1,ngram_range=(1,1))
            tfidf = headline_vectorizer.fit_transform(trainContent)

            if (method_cl == 'dbscan'):
                x_object = getDBSCAN(train_tweets,tfidf,epsilon)
            elif (method_cl == 'hdbscan'):
                x_object = getHDBSCAN(train_tweets,tfidf)
            elif (method_cl == 'hierarchical'):
                x_object = getHierarchical(train_tweets,tfidf)
            elif (method_cl == 'kmeans'):
                x_object = getKMeans(train_tweets,tfidf)
            else:
                x_object = getAffinity(train_tweets,tfidf)

            candidate_cluster = x_object["clusters"]
            if (method_cl=='dbscan' or method_cl=='hdbscan'):
                candidate_noise = x_object["noise_data"]
                for index,row in candidate_noise.iterrows():
                    list_y.append(row["Event"])
                    list_pr.append(-1)
                    list_y_fscore.append(row["Event"])
                    list_pr_fscore.append(-1)
            candidate_labels= x_object["labels_pr"]

            for current_cluster in candidate_cluster:
                pr_event = getLabel(current_cluster["Event"])

                #Create Map Dataframe
                temp_df = current_cluster[['tweets','coordinates']]
                temp_df['Timestamp'] = current_date
                temp_df['Cluster'] = count_labels
                temp_df['Event'] = np.array([temp for temp in current_cluster['Event']])
                return_df = pd.concat([return_df,temp_df])

                #Create Cluster's specs dataframe
                specs_cluster_df = specs_cluster_df.append({'Cluster_ID': count_labels, 'Cluster_Length': current_cluster.shape[0],
                                                            'Predicted_Event': int(pr_event),'Purity' : getPurity(current_cluster,pr_event),
                                                            'Timestamp' : current_date}, ignore_index=True)

                for index,row in current_cluster.iterrows():
                    list_y.append(row["Event"])
                    list_pr.append(count_labels)
                    list_y_fscore.append(row["Event"])
                    list_pr_fscore.append(pr_event)
                count_labels+=1

            from sklearn import metrics
            adjusted_mutual_info.append(metrics.adjusted_mutual_info_score(list_y,list_pr,average_method='arithmetic'))
            adjusted_rand_score.append(metrics.adjusted_rand_score(list_y,list_pr))
            completeness_score.append(metrics.completeness_score(list_y,list_pr))
            fowlkes_mallows_score.append(metrics.fowlkes_mallows_score(list_y,list_pr,sparse=False))
            homogeneity_score.append(metrics.homogeneity_score(list_y,list_pr))
            mutual_info_score.append(metrics.mutual_info_score(list_y,list_pr,contingency=None))
            normalized_mutual_info_score.append(metrics.normalized_mutual_info_score(list_y,list_pr,average_method='arithmetic'))
            v_measure_score.append(metrics.v_measure_score(list_y,list_pr))
        except Exception as e:
            print(e)
            pass
    from sklearn import metrics
    f1 = metrics.f1_score(list_y_fscore,list_pr_fscore,average='micro')
    clustering_name = ''
    num_grams = '('+str(ngram)+',3)'
    details = {'clustering_name':method_cl,'num_grams':num_grams,'epsilon':epsilon,
                   'min_df':minimum,'max_df':maximum}

    try:
        clustering_object =  {'f1_score':f1,'adjusted_mutual_info_score':statistics.mean(adjusted_mutual_info),'adjusted_rand_score':statistics.mean(adjusted_rand_score),
                                'completeness_score':statistics.mean(completeness_score),'fowlkes_mallows_score':statistics.mean(fowlkes_mallows_score),
                                'homogeneity_score':statistics.mean(homogeneity_score),'mutual_info_score':statistics.mean(mutual_info_score),
                                'normalized_mutual_info_score':statistics.mean(normalized_mutual_info_score),
                                 'v_measure_score':statistics.mean(v_measure_score)}

        total_result = {'parameters':details,'clustering_object':clustering_object}
        return json.dumps(total_result),return_df,specs_cluster_df
    except Exception as e:
        print(e)
        return None
    return None


def create_clusters_show_time_window(df,ngram,epsilon,mindf,maxdf,coordinates,method_cl,list_date_minutes):
    temporal_clusters = None
    saved_clusters = None
    df['Clustered'] = False
    df['Cluster'] = -1
    count_labels = 0
    timestamp = 0
    count_timestamp = 1000
    list_cluster_timer = []
    list_cluster_cycles = []
#     for k in range(0,len(list_date_minutes)-10):
    for k in range(20):
        timestamp+=1
        if timestamp == count_timestamp:
            print('Timestamp:',timestamp)
            count_timestamp+=1000
        for i in range(len(list_cluster_timer)):
            list_cluster_timer[i]+=1
            list_cluster_cycles[i] += 1
        df1 = df.loc[df['Datetime']==list_date_minutes[k]]
        df2 = df.loc[df['Datetime']==list_date_minutes[k+1]]
        df3 = df.loc[df['Datetime']==list_date_minutes[k+2]]
        df4 = df.loc[df['Datetime']==list_date_minutes[k+3]]
        df5 = df.loc[df['Datetime']==list_date_minutes[k+4]]
        df6 = df.loc[df['Datetime']==list_date_minutes[k+5]]
        df7 = df.loc[df['Datetime']==list_date_minutes[k+6]]
        df8 = df.loc[df['Datetime']==list_date_minutes[k+7]]
        df9 = df.loc[df['Datetime']==list_date_minutes[k+8]]
        df10 = df.loc[df['Datetime']==list_date_minutes[k+9]]
        list_of_dataframes = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,temporal_clusters]
        dfchange = pd.concat(list_of_dataframes)
        dfchange = dfchange.sort_values(by=['Datetime'])

        tweets = dfchange.sort_values(by=['time'])
        if coordinates == False:
            tweetsContent = tweets.copy()["clean_data"]
        else:
            tweetsContent = tweets.copy()['clean_data_coordinates']

        tfidf_object = cl.getTfIdf(tweetsContent,dfchange.shape[0],ngram,3,mindf,maxdf)
        tfidf = tfidf_object['tfidf_train_data_features']
        minimum = tfidf_object['minimum']
        maximum = tfidf_object['maximum']
        list_sum = tfidf_object['sum']
        tweets['sum'] = list_sum
        noise_tweets = tweets.loc[tweets['sum']==0]

#         for index,row in noise_tweets.iterrows():
#             list_y_fscore.append(row["Event"])
#             list_pr_fscore.append(-1)

        train_tweets = tweets.loc[tweets['sum']!=0]
        if coordinates == False:
            trainContent = train_tweets.copy()["clean_data"]
        else:
            trainContent = train_tweets.copy()['clean_data_coordinates']

        headline_vectorizer = CountVectorizer(binary=True, min_df=1,ngram_range=(1,1))
        tfidf = headline_vectorizer.fit_transform(trainContent)

        if (method_cl == 'dbscan'):
            x_object = cl.getDBSCAN(train_tweets,tfidf,epsilon)
        elif (method_cl == 'hdbscan'):
            x_object = cl.getHDBSCAN(train_tweets,tfidf)
        elif (method_cl == 'hierarchical'):
            x_object = cl.getHierarchical(train_tweets,tfidf)
        elif (method_cl == 'kmeans'):
            x_object = cl.getKMeans(train_tweets,tfidf)
        else:
            x_object = cl.getAffinity(train_tweets,tfidf)

        candidate_cluster = x_object["clusters"]
        if (method_cl=='dbscan' or method_cl=='hdbscan'):
                candidate_noise = x_object["noise_data"]

        new_candidates = []
        for temp_cluster in candidate_cluster:
            new_candidates.append(temp_cluster)

        for temp in new_candidates:
            temp_cluster = temp.loc[temp['Clustered']==True]
            if temp_cluster.shape[0]==0:
                df = df.drop(temp.index.tolist())
                temp['Clustered'] = True
                temp['Cluster'] = count_labels
                list_cluster_timer.append(0)
                list_cluster_cycles.append(0)

                count_labels+=1
                temporal_clusters = pd.concat([temporal_clusters,temp])
            else:
                label = getClusterLabel(temp_cluster)
                label_cluster = temporal_clusters.loc[temporal_clusters['Cluster']==label]
                if temp.shape[0] != label_cluster.shape[0]:
#                     new_items = temp.loc[temp['Clustered']==False]
                    new_items = temp.loc[temp['Cluster']!=label]
                    df = df.drop(new_items.index.tolist())
                    temporal_clusters = temporal_clusters.drop(new_items.index.tolist())
                    new_items['Clustered'] = True
                    new_items['Cluster'] = label
                    temporal_clusters = pd.concat([temporal_clusters,new_items])
                    list_cluster_timer[label] = 0

        for i in range(len(list_cluster_timer)):
            if(list_cluster_timer[i]==3 or list_cluster_cycles[i]==10):
#                 import pdb; pdb.set_trace()
                temp = temporal_clusters.loc[temporal_clusters['Cluster']==i]
                if temp.shape[0]!=0:
                    temp['Timestamp'] = timestamp
                    temp_df = temp[['tweets','coordinates','Timestamp','Cluster','Event']]
                    saved_clusters = pd.concat([saved_clusters,temp_df])
                    print('Timestamp:',list_date_minutes[k],'-',list_date_minutes[k+9])
                    print('Size: ',temp.shape[0])
                    for index,row in temp.iterrows():
                        print(row['Event'],' ',end='')
                    print()
                    print()
                temporal_clusters = temporal_clusters.drop(temp.index.tolist())
                list_cluster_timer[i]=-1
                list_cluster_cycles[i] = -1
    return saved_clusters

def create_clusters_show_hourly(df,ngram,epsilon,mindf,maxdf,method_cl,list_date_hour):
    count = 0
    count_labels = 0
#     for current_date in list_date_minutes[:-10]:
    for k in range(0,len(list_date_minutes)-10):
        try:
            list_y = []
            list_pr = []

            dfchange = df.loc[df['DateHour']==current_date]
            tweets = dfchange.sort_values(by=['time'])
            tweetsContent = tweets.copy()["clean_data"]
            tfidf_object = cl.getTfIdf(tweetsContent,dfchange.shape[0],ngram,3,mindf,maxdf)
            tfidf = tfidf_object['tfidf_train_data_features']
            minimum = tfidf_object['minimum']
            maximum = tfidf_object['maximum']
            list_sum = tfidf_object['sum']
            tweets['sum'] = list_sum
            noise_tweets = tweets.loc[tweets['sum']==0]

            print("Event",count,":")
            count+=1
            print("Cluster_noise: ",noise_tweets.shape[0],"False Negative:",noise_tweets.loc[noise_tweets["Event"]!=-1].shape[0])

            train_tweets = tweets.loc[tweets['sum']!=0]
            print("Tweets: ",train_tweets.shape[0])
            trainContent = train_tweets.copy()["clean_data"]

#             import pdb; pdb.set_trace()
            headline_vectorizer = CountVectorizer(binary=True, min_df=1,ngram_range=(1,1))
            tfidf = headline_vectorizer.fit_transform(trainContent)

            if (method_cl == 'dbscan'):
                x_object = getDBSCAN(train_tweets,tfidf,epsilon)
            elif (method_cl == 'hdbscan'):
                x_object = getHDBSCAN(train_tweets,tfidf)
            elif (method_cl == 'hierarchical'):
                x_object = getHierarchical(train_tweets,tfidf)
            elif (method_cl == 'kmeans'):
                x_object = getKMeans(train_tweets,tfidf)
            else:
                x_object = getAffinity(train_tweets,tfidf)

            candidate_cluster = x_object["clusters"]
            if (method_cl=='dbscan' or method_cl=='hdbscan'):
                candidate_noise = x_object["noise_data"]
                candidate_noise_not = candidate_noise.loc[candidate_noise["Event"]!=-1]
                print("Total new noise:",candidate_noise.shape[0],"New False Negative:",candidate_noise_not.shape[0])
                for index,row in candidate_noise_not.iterrows():
                    print(row["Event"]," ",end='')
                print()


            candidate_labels= x_object["labels_pr"]

            print("Number of Clusters:",len(candidate_cluster))
            for current_cluster in candidate_cluster:
                print("Cluster Label:",count_labels,current_cluster.shape[0])
                pr_event = getLabel(current_cluster["Event"])
                for index,row in current_cluster.iterrows():
                    print(row["Event"]," ",end='')
                print()
                count_labels+=1

        except Exception as e:
            print(e)
            pass
