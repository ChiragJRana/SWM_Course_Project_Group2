import sklearn
import pandas as pd
from sklearn.cluster import KMeans
import pickle


def load_clusters():
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans_clusters = pickle.load(f)
    return kmeans_clusters


def get_recommendations(user_id):
    # Get user clusters
    user_clusters = kmeans.predict(pivot_df)

    # Recommend movies for each user
    recommended_movies = []

    # Get user cluster
    user_cluster = user_clusters[user_id]

    # Find closest user in the same cluster
    closest_user_id = pivot_df[kmeans.labels_ == user_cluster].index[0]

    # Get movies rated highly by closest user but not by target user
    top_rated_movies = pivot_df.loc[closest_user_id][pivot_df.loc[user_id] == 0].sort_values(
        ascending=False)

    # Get movies rated by target user
    target_user_movies = pivot_df.loc[user_id][pivot_df.loc[user_id] != 0].index

    # remove common movies
    movies = top_rated_movies.index.difference(target_user_movies)

    # Shuffle the index
    shuffled_movies = movies.to_series().sample(frac=1).index

    # Get top recommended movies
    recommended_movies.extend(shuffled_movies[:5])

    return recommended_movies


def get_movie_recommendations(user_id):
    recommended_movies = get_recommendations(user_id)
    recommended_movies = [eval(i) for i in recommended_movies]
    movies = ratings_df.loc[ratings_df['movie_id'].isin(recommended_movies)]
    movies.drop_duplicates(subset='movie_id', keep="first", inplace=True)
    return movies


def get_temp_prediction(user_id):
    recommended_movies = [13509, 1475, 13708, 10261, 7378]
    movies = ratings_df.loc[ratings_df['movie_id'].isin(recommended_movies)]
    movies.drop_duplicates(subset='movie_id', keep="first", inplace=True)
    return movies.head()


kmeans = load_clusters()
pivot_df = pd.read_csv('pivot_df.csv')
pivot_df.drop(pivot_df.columns[[0]], axis=1, inplace=True)
ratings_df = pd.read_csv('netflix_filtered_ratings.csv')
