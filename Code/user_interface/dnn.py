import pandas as pd
import numpy as np
import pickle


def load_model():
    # Load the trained model
    model_file = 'dnn.pkl'
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    return model


def get_movie_recommendations(user_id):
    # Get the movies not rated by the user
    user_rated_movies = movie_data[movie_data['user_id'] == user_id]['movie_id']
    all_movies = np.arange(len(movie_data.movie_id.unique()))
    movies_to_recommend = np.setdiff1d(all_movies, user_rated_movies)

    user_ids = np.full(len(movies_to_recommend), user_id)
    recommendation_data = pd.DataFrame({'user_id': user_ids, 'movie_id': movies_to_recommend})
    # Get the predicted ratings for the movies to recommend
    X_test = [recommendation_data.user_id, recommendation_data.movie_id]
    y_pred = model.predict(X_test)

    recommendation_data['predicted_rating'] = y_pred

    recommendation_data = recommendation_data.sort_values(by=['predicted_rating'], ascending=False)
    # Get the top 5 recommendations
    top_recommendations = recommendation_data.head(5)

    recommendation_list = []

    for index, row in top_recommendations.iterrows():
        movie_id = row['movie_id']
        movie_title = movie_data[movie_data['movie_id'] == movie_id]['movie_title'].iloc[0]
        predicted_rating = row['predicted_rating']
        recommendation_list.append({
            'Movie ID': movie_id,
            'Movie Title': movie_title,
            'Predicted Rating': round(predicted_rating, 2)
        })
    return recommendation_list

model = load_model()
# Load the movie data
movie_data = pd.read_csv(
    '../../../../Downloads/SWM_Course_Project_Group2-DL/SWM_Course_Project_Group2-DL/dnn_movie_list.csv', usecols=['user_id', 'movie_id', 'movie_title'], dtype={'user_id': 'int32', 'movie_id': 'int16', 'movie_title': 'object'})
movie_data['user_id'] = movie_data['user_id'].astype('category').cat.codes.values
movie_data['movie_id'] = movie_data['movie_id'].astype('category').cat.codes.values