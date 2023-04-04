#Reference: https://www.kaggle.com/code/danofer/deep-learning-for-netflix-prize-challenge/notebook
import os
import pandas as pd

if not os.path.isfile('data.csv'):
    data = open('data.csv', mode='w')

files = ['dataset/combined_data_1.txt',
         'dataset/combined_data_2.txt',
          'dataset/combined_data_3.txt',
          'dataset/combined_data_4.txt'
        ]

# Remove the line with movie_id: and add a new column of movie_id
# Combine all data files into a csv file
for file in files:
  print("Opening file: {}".format(file))
  with open(file) as f:
    for line in f:
        line = line.strip()
        if line.endswith(':'):
            movie_id = line.replace(':', '')
        else:
            data.write(movie_id + ',' + line)
            data.write('\n')
data.close()

# Read all data into a pd dataframe
df = pd.read_csv('data.csv', names=['movie_id', 'user_id','rating','date'])
print(df.head())
