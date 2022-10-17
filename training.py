import pandas as pd
import numpy as np
import os
import datetime
from collections import Counter
import queue
import threading
import time

class DataManager:
    def __init__(self):
        self.df_queue = queue.Queue(maxsize=0)
        self.threads = []

    def start_threads(self,file, fn,threads=1,lines=50000):
        for i in range(threads):
            insert_fn = lambda x: self.df_queue.put(x)
            self.threads.append(threading.Thread(target=load_anime_lists,args=(file,i,fn,insert_fn,lines),daemon=True))
            self.threads[-1].start()

    def stop_threads(self):
        for t in self.threads:
            t.join()
        self.threads = []  

def process_data(df):
    print(df.head())
    df = df[["username","anime_id","my_watched_episodes","my_score","my_status","my_last_updated"]]
    df.loc[:,'unrated'] = df["my_score"].map(lambda i: int(i==0))
    df.loc[:,'my_status'] = df["my_status"].map(lambda i: i if i<6 else 5)
    df.loc[:,'year'] = df["my_last_updated"].map(lambda x: datetime.datetime.fromtimestamp(x).year)
    return df

def load_anime_lists(file:str,iteration:int,fn,insert_fn,lines=500000,sort=True):
    t = time.time()
    print("started ",iteration)
    df = pd.read_csv(file,skiprows=range(1,iteration*lines),nrows=lines)
    if len(df) == 0:
        return
    print(df)
    df = fn(df) 
    print(df)
    # convert list with names into list of lists with unique names
 
    while not df.empty:
        print(len(df))
        name = df["username"].iloc[0]
        section = df[df["username"] == name]
        if sort:
            section = section.sort_values(by="my_last_updated")
        insert_fn(section)
        df = df.loc[df["username"] != name]
    
    '''
    lastuser = df["username"].iloc[1]
    list_size = 0
    for index, row in df.iterrows():
        if row["username"] == lastuser:
            list_size+=1
        else:
            if list_size< 400 and list_size>0:
                #print(index)
                section = df.loc[index-1-list_size:index-1]
                if sort:
                    section = section.sort_values(by="my_last_updated")
                insert_fn(section)
                #print(list_size)
                #print(df.loc[index-1-list_size:index-1])

            #print(row["username"],list_size)
            lastuser = row["username"]
            list_size=0
    
    '''
    print("finished ",iteration, " in ",time.time()-t)

def save_process_data(new_file:str,old_file:str,fn,lines=500000):
    dfs = []
    insert_fn = lambda i: dfs.append(i)
    load_anime_lists(old_file,0,fn,insert_fn,lines,True)

    df = pd.DataFrame()
    for i in dfs:
        df = pd.concat([df,i])

    print(df)
    df.to_csv(new_file)



def load_animes(file):
    df = pd.read_csv(file)
    df = df[["title","anime_id","type","source","episodes","duration_min","rating","score","scored_by","rank","popularity","members","favorites","genre","aired_from_year","studio"]]

    for i in ["type","rating","source"]:
        print(set(df[i]))
    studios = [x.split(", ") for x in list(df["studio"])]
    freq = Counter(sum(studios,[]))
    #number of animes under a studio to be considered individually
    cutoff = 15
    biggest_studios = list(filter(lambda x:x[1]>cutoff,freq.items()))

    print(biggest_studios,len(biggest_studios),sum([x[1] for x in biggest_studios]))

    other_studios = list(filter(lambda x:x[1]<=cutoff,freq.items()))
    other_studios = [x[0] for x in other_studios],sum([x[1] for x in other_studios])
    genres = [str(x).split(", ") for x in list(df["genre"])]

    genres_freq = Counter(sum(genres,[]))
    #print(genres_freq)
    #print(other_studios)
    return df, biggest_studios, other_studios, genres_freq


    
if __name__ == "__main__":
    df, studios, other_studios, genres = load_animes(os.path.join("data","anime_cleaned.csv"))

    anime_ids = {v:k+1 for k,v in df["anime_id"].sort_values().to_dict().items()}
    backwards_anime_ids = {k+1:v for k,v in df["anime_id"].to_dict().items()}
    anime_lookup = dict(zip(df["anime_id"],df["title"]))
    print(df["anime_id"].to_dict())
    dfs = []
    insert_fn = lambda i: dfs.append(i)
    load_anime_lists(os.path.join("new_data","animelist.csv"),0,fn=lambda df: df.loc[df["my_last_updated"]>0],insert_fn=insert_fn,sort=False)
    print(dfs[0])
'''
    def new_process_data(df):
        df = process_data(df)
        df.loc[:,"anime_id"] = df["anime_id"].map(lambda i: anime_ids[i] if i in anime_ids else 0)
        return df
    
    save_process_data(os.path.join("new_data","animelist.csv"),os.path.join("data","UserAnimeList.csv"),new_process_data,5000000)
    
    manager = DataManager()
    manager.start_threads(os.path.join("data","UserAnimeList.csv"),process_data,1,800000)
    manager.stop_threads()
    '''
    #dfs = load_anime_lists(os.path.join("data","UserAnimeList.csv"),0,fn=process_data,lines=20000)


    