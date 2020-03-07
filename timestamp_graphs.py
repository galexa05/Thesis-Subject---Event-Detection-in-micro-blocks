import datetime
from time import strptime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def getList(df,name):
    tableFlag=[]
    data_sorted = df.sort_values(by=['date'],inplace=False)
    data_sorted = data_sorted.reset_index(drop=True)
    list_date = []
    for temp in data_sorted[name]:
        list_date.append(temp)
    list_date = list(dict.fromkeys(list_date))
    list_date = sorted(list_date)
    return list_date

def create_date(text):
    temp = text.split(" ")
    x = str(datetime.datetime(int(temp[5]),int(strptime(temp[1],'%b').tm_mon),int(temp[2]))).split()[0]
    return x

def createTimestamps(df):
    df['date'] = df['created_at'].apply(lambda x: create_date(x))
    df['time'] =  np.array([tweet.split()[3] for tweet in df["created_at"]])
    df['Datetime']= pd.to_datetime(df['date'].apply(str)+' '+df['time'].apply(lambda x: x.split(':')[0]) + df['time'].apply(lambda x: x.split(':')[1]))
    df['DateHour'] = pd.to_datetime(df['date'].apply(str)+' '+df['time'].apply(lambda x: x.split(':')[0])+':00')
    df['Date_Ten_Minutes'] = pd.to_datetime(df['date'].apply(str)+' '+df['time'].apply(lambda x: x.split(':')[0])+':'+df["time"].apply(lambda x: x.split(":")[1][0]+'0'))
    return df

def monitorGraphPerTenMinutes(df_relevant,df_irrelevant):
    # graphing the threshold vs hourly tweet occurences
    dfchange = df_irrelevant.loc[df_irrelevant['date'] != -1]
    ts = dfchange.set_index('Date_Ten_Minutes')
    vc = ts.groupby('Date_Ten_Minutes').count()
    col = ['id']
    vc2 = vc[col]
    vc3 = vc2.copy()


    dfchange1 = df_relevant.loc[df_relevant["date"]!=-1]
    rs = dfchange1.set_index('Date_Ten_Minutes')
    rc = rs.groupby('Date_Ten_Minutes').count()
    col = ['id']
    rc2 = rc[col]
    rc3 = rc2.copy()

    vc3.rename(columns={'id':'Irrelevant Tweets'},inplace=True)

    rc3.rename(columns={'id':'Relevant Tweets'},inplace=True)

    ax = vc3.plot()
    rc3.plot(ax = ax)

def monitorGraphPerMinute(df_relevant,df_irrelevant):
    # gca stands for 'get current axis'
    # graphing the threshold vs hourly tweet occurences
    dfchange = df_irrelevant.loc[df_irrelevant['date'] != -1]
    ts = dfchange.set_index('Datetime')
    vc = ts.groupby('Datetime').count()
    col = ['id']
    vc2 = vc[col]
    vc3 = vc2.copy()


    dfchange1 = df_relevant.loc[df_relevant["date"]!=-1]
    rs = dfchange1.set_index('Datetime')
    rc = rs.groupby('Datetime').count()
    col1 = ['id']
    rc2 = rc[col1]
    rc3 = rc2.copy()

    vc3.rename(columns={'id':'Noise Tweets'},inplace=True)
    rc3.rename(columns={'id':'Relevant Tweets'},inplace=True)

    ax = vc3.plot()
    rc3.plot(ax = ax)

def monitorGraphPerHour(df_relevant,df_irrelevant):
    # gca stands for 'get current axis'
    # graphing the threshold vs hourly tweet occurences
    dfchange = df_irrelevant.loc[df_irrelevant['date'] != -1]
    ts = dfchange.set_index('DateHour')
    vc = ts.groupby('DateHour').count()
    col = ['id']
    vc2 = vc[col]
    vc3 = vc2.copy()


    dfchange1 = df_relevant.loc[df_relevant["date"]!=-1]
    rs = dfchange1.set_index('DateHour')
    rc = rs.groupby('DateHour').count()
    col1 = ['id']
    rc2 = rc[col1]
    rc3 = rc2.copy()

    vc3.rename(columns={'id':'Noise Tweets'},inplace=True)
    rc3.rename(columns={'id':'Relevant Tweets'},inplace=True)

    ax = vc3.plot()
    rc3.plot(ax = ax)
