import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# === Load Data ===

# Ratings
ratings = pd.read_csv(
    r'C:\Users\ankis\Downloads\ak\ml-100k\u.data',
    sep='\t',
    names=['user_id', 'item_id', 'rating', 'timestamp']
)

# Movies
movie_columns = [
    "item_id", "title", "release_date", "video_release_date", "IMDb_URL",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
movies = pd.read_csv(
    r'C:\Users\ankis\Downloads\ak\ml-100k\u.item',
    sep='|',
    encoding='latin-1',
    header=None,
    names=movie_columns
)

# Merge data for convenience
movie_titles = movies[['item_id', 'title']]
user_item_matrix = ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# === Content-Based Filtering ===

# Combine genre columns into a single string per movie
genre_cols = movie_columns[5:]
movies['genres'] = movies[genre_cols].apply(
    lambda row: ' '.join([genre for genre, val in row.items() if val == 1]),
    axis=1
)

# TF-IDF and cosine similarity
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title'])

def content_recommend(title, top_n=5):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i for i, _ in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# === Collaborative Filtering using SVD (scikit-learn) ===

svd = TruncatedSVD(n_components=20, random_state=42)
matrix_reduced = svd.fit_transform(user_item_matrix)
predicted_ratings = np.dot(matrix_reduced, svd.components_)
predicted_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)

def svd_recommend(user_id, top_n=5):
    if user_id not in predicted_df.index:
        return []
    user_row = predicted_df.loc[user_id]
    rated_items = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    recommendations = user_row.drop(index=rated_items).sort_values(ascending=False).head(top_n)
    return recommendations.index.tolist()

# === Streamlit App ===

st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎬 Personalized Movie Recommendation System")

# Select user and movie
user_id = st.selectbox("Select your User ID", sorted(ratings['user_id'].unique()))
movie = st.selectbox("Choose a movie you like", sorted(movies['title'].unique()))

# Content-based results
st.subheader("📽️ Content-Based Recommendations")
content_recs = content_recommend(movie)
if content_recs:
    for rec in content_recs:
        st.write("•", rec)
else:
    st.write("No similar movies found.")

# Collaborative results
st.subheader("🤝 Collaborative Recommendations")
collab_ids = svd_recommend(user_id)
if collab_ids:
    for rec_id in collab_ids:
        rec_title = movies[movies['item_id'] == rec_id]['title'].values[0]
        st.write("•", rec_title)
else:
    st.write("No collaborative recommendations found.")
