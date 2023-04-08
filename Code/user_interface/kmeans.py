import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import pickle

def load_clusters():
    with open('kmeans_clusters.pkl', 'rb') as f:
      kmeans_clusters = pickle.load(f)
    return kmeans_clusters

def load_movies_clusters():
    with open('kmeans_movies_clusters.pkl', 'rb') as f:
      kmeans_movie_clusters = pickle.load(f)
    return kmeans_movie_clusters

def recommend_movies(userId):
	recommendations = make_recommendations(userId, movie_rating_above_4).recommend_new_movies()[1]
	return movies_data[movies_data['movieId'].isin(recommendations[:5])]

def get_user_movies(user_id, users_data):
    return list(users_data[users_data['userId'] == user_id]['movieId'])

class make_recommendations:
    def __init__(self, user_id, users_data):
        self.users_data = users_data.copy()
        self.user_id = user_id
        users_cluster = clusters
        self.user_cluster = int(users_cluster[users_cluster['userId'] == self.user_id]['Cluster'])
        self.movies_list = movies_clusters
        self.cluster_movies = self.movies_list[self.user_cluster]
        self.cl_mv_list = list(self.cluster_movies['movieId'])

    def recommend_new_movies(self):
        try:
            user_movies = get_user_movies(self.user_id, self.users_data)
            cl_mv_list = self.cl_mv_list.copy()
            for um in user_movies:
                if um in cl_mv_list:
                    cl_mv_list.remove(um)
            return [True, cl_mv_list]
        except KeyError:
            err = "User history does not exist"
            print(err)
            return [False, err]
        except:
            err = 'Error: {0}, {1}'.format(exc_info()[0], exc_info()[1])
            print(err)
            return [False, err]


movie_rating_above_4 = pd.read_csv('filtered_ratings.csv', usecols = ['userId', 'movieId','rating'])
movies_data = pd.read_csv('movie.csv', usecols = ['movieId', 'genres', 'title'])
clusters = load_clusters()
movies_clusters = load_movies_clusters()