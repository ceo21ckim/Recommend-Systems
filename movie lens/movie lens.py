import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
import os, re 

# checking dir
os.getcwd()

path = r'C:\Users\EonKim\Desktop\경희대\books\python\dataset\movielens'
os.chdir(path)

file_list = os.listdir()
file_list


links = pd.read_csv(file_list[0])
movies = pd.read_csv(file_list[1])
ratings = pd.read_csv(file_list[2])
tags = pd.read_csv(file_list[-1])


# EDA 
links.info() # movieId, imdbId, tmdbId 
links.head()

movies.info() # movieId, title, genres
movies.head()

ratings.info() # userId, movieId, rating, timestamp
ratings.head()

tags.info() # userId, movieId, tag, timestamp
tags.head()

df = pd.merge(ratings, movies, how = 'inner')

df['year'] = df['title'].map(lambda x: re.sub('[^\d{4}]', '', x))
df['title'] = df['title'].map(lambda x: x[:-7])
df['genres'] = df['genres'].str.replace('|', ' ')


# making pivot_table
table = pd.pivot_table(data = df, index = 'userId', columns = 'title', values = 'rating')


# unstack() , stack()
user = []
title = []
for i in range(len(table)):
    user += [table.stack().index[i][0]]
    title += [table.stack().index[i][1]]

answer = pd.DataFrame()
answer['userId'] = user 
answer['title'] = title







# correlaction 
user_item_corr = table.corr()

corr_set = user_item_corr.stack().sort_values( ascending= False)
corr_set = round(corr_set, 4)

c = corr_set[corr_set != 1]



def recsys(title, k):
    df = pd.merge(ratings, movies, how = 'inner')

    df['year'] = df['title'].map(lambda x: re.sub('[^\d{4}]', '', x))
    df['title'] = df['title'].map(lambda x: x[:-7])
    df['genres'] = df['genres'].str.replace('|', ' ')

    table = pd.pivot_table(data = df, index = 'userId', columns = 'title', values = 'rating')

    user_item_corr = table.corr()

    target = round(user_item_corr.stack().sort_values( ascending= False), 6)

    target = target[target != 1]

    t1, t2 = [], []
    for i, movie in enumerate(target.index):
        t1.append(movie[0])
        t2.append(movie[1])
    ans = dict({
        'movie1' : t1,
        'movie2' : t2, 
        'correlation' : target.values.tolist()
        })
    answer = pd.DataFrame(ans)

    return answer[answer['movie1'] == title][:k]

ti = 'Hours, The'
recsys(ti, 5)
