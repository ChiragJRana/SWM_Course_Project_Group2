import sklearn
import pandas as pd
from sklearn.cluster import KMeans
import pickle

user_specific = pd.read_csv('user_specific.csv')
ratings_df = pd.read_csv('netflix_filtered_ratings.csv')

def load_model():
    with open('svd.pkl', 'rb') as f:
      svd = pickle.load(f)
    return svd

def get_recommendations(user_id, user_specific):
    user_specific['Estimate_Score'] = user_specific['Movie_Id'].apply(lambda x: svd.predict(user_id, x).est)
    user_specific = user_specific.sort_values('Estimate_Score', ascending=False)
    return user_specific.head()

def get_movie_recommendations(user_id):
  recommended_movies = get_recommendations(user_id, user_specific)
  recommended_movies = recommended_movies.rename(columns={
    'Movie_Id': 'movie_id', 
    'Estimate_Score': 'movie_rating', 
    'Name': 'movie_title'
    })
  return recommended_movies

svd = load_model()
