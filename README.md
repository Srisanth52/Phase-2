import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval

movies_df = pd.read_csv("tmdb_5000_movies.csv")


movies_df['overview'] = movies_df['overview'].fillna('')
movies_df['genres'] = movies_df['genres'].apply(literal_eval)


def create_soup(x):
    return ' '.join([genre['name'] for genre in x['genres']]) + ' ' + x['overview']

movies_df["soup"] = movies_df.apply(create_soup, axis=1)


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['soup'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()


def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return f"Movie '{title}' not found in database."
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]


if _name_ == "_main_":
    user_input = input("Enter a movie name: ")
    recommendations = get_recommendations(user_input)
   
    if isinstance(recommendations, str):
        print(recommendations)
    else:
        print(f"\nRecommendations for '{user_input}':")
        for title in recommendations:
            print(title)
