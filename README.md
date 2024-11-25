# project-python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample movie dataset
movies_data = {
    'movieId': [1, 2, 3, 4, 5],
    'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'genres': ['Action|Adventure', 'Action|Sci-Fi', 'Adventure|Fantasy', 'Drama|Romance', 'Comedy|Romance']
}

ratings_data = {
    'userId': [1, 1, 2, 2, 3, 3, 3],
    'movieId': [1, 2, 1, 3, 2, 4, 5],
    'rating': [5, 4, 5, 3, 2, 5, 4]
}

movies = pd.DataFrame(movies_data)
ratings = pd.DataFrame(ratings_data)

# Merging movies and ratings
movie_ratings = pd.merge(ratings, movies, on='movieId')

# Creating a pivot table
pivot_table = movie_ratings.pivot_table(index='userId', columns='title', values='rating')

# Fill NaN with 0 for calculations
pivot_table = pivot_table.fillna(0)

# Compute cosine similarity
similarity_matrix = cosine_similarity(pivot_table)
similarity_df = pd.DataFrame(similarity_matrix, index=pivot_table.index, columns=pivot_table.index)

def get_movie_recommendations(movie_title, user_id, num_recommendations=3):
    # Get the index of the movie that matches the title
    if movie_title not in pivot_table.columns:
        return "Movie not found."

    movie_idx = pivot_table.columns.get_loc(movie_title)

    # Get the user's ratings
    user_ratings = pivot_table.loc[user_id]

    # Get similarity scores for the movie
    sim_scores = similarity_df[movie_idx]

    # Create a DataFrame for the scores
    score_series = pd.Series(sim_scores, index=pivot_table.index)
    
    # Exclude the current user and sort the scores
    score_series = score_series.drop(user_id).sort_values(ascending=False)

    # Get top n recommendations
    recommended_user_ids = score_series.head(num_recommendations).index

    # Get recommended movie titles
    recommended_movies = pivot_table.loc[recommended_user_ids].mean().sort_values(ascending=False)

    return recommended_movies.index.tolist()

# Example usage
user_id = 1
movie_title = 'Movie A'
recommendations = get_movie_recommendations(movie_title, user_id)

print(f"Recommendations for user {user_id} based on '{movie_title}':")
print(recommendations)

