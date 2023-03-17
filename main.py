#Reference: https://www.kaggle.com/code/sreehariis/recommender-system

import pandas as pd
import numpy as np

dataset = pd.read_csv('dataset/combined_data_1.txt', header=None, names=['Customer_Id', 'Rating'], usecols=[0, 1])
print(dataset.head())
print(dataset.dtypes)

df_nan = pd.DataFrame(pd.isnull(dataset.Rating))
df_nan = df_nan[df_nan['Rating'] == True]
df_nan=df_nan.reset_index()

movie_np = []
movie_id = 1

for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
    temp = np.full((1,i-j-1), movie_id)
    movie_np = np.append(movie_np, temp)
    movie_id += 1

# Account for last record and corresponding length
last_record = np.full((1,len(dataset) - df_nan.iloc[-1, 0] - 1),movie_id)
movie_np = np.append(movie_np, last_record)


#dataset[pd.notnull(dataset['Rating'])]
dataset = dataset[pd.notnull(dataset['Rating'])]

dataset['Movie_Id'] = movie_np.astype(int)
dataset['Customer_Id'] =dataset['Customer_Id'].astype(int)
print('-Dataset examples-')
print(dataset.head())
