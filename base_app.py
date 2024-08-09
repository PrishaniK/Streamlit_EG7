import numpy as np
import streamlit as st
import pandas as pd
import pickle
from surprise import Reader, Dataset
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


@st.cache_resource
def load_image(image_path):
    return Image.open(image_path)

@st.cache_data
def load_anime_data():
    return pd.read_csv('anime.csv')

@st.cache_data
def load_ratings_data():
    return pd.read_csv('train.csv')

@st.cache_resource
def load_svd_model():
    with open('svd_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def compute_tfidf_matrix(anime_data):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    return tfidf_vectorizer.fit_transform(anime_data['genre'])

@st.cache_resource
def compute_cosine_similarity(_tfidf_matrix):
    return cosine_similarity(_tfidf_matrix, _tfidf_matrix)

# Load data
anime_data = load_anime_data()
ratings = load_ratings_data()

# Ensure required columns are present and fill NaNs with empty strings
if 'name' in anime_data.columns and 'genre' in anime_data.columns:
    anime_data['name'] = anime_data['name'].fillna('')
    anime_data['genre'] = anime_data['genre'].fillna('')
else:
    st.error("The required columns ('name' and 'genre') are not present in the dataset.")
    st.stop()

# Load models
svd_model = load_svd_model()

# Compute TF-IDF matrix and cosine similarity
tfidf_matrix = compute_tfidf_matrix(anime_data)
cosine_sim = compute_cosine_similarity(tfidf_matrix)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Recommend Anime", "Overview", "Insights", "Anime Archive", "About Us"], key="navigation")

def get_content_based_recommendations(anime_id=None, anime_name=None, num_recommendations=10):
    if anime_id is not None:
        idx = anime_data.index[anime_data['anime_id'] == anime_id].tolist()
    elif anime_name is not None:
        idx = anime_data.index[anime_data['name'].str.contains(anime_name, case=False, na=False)].tolist()
    else:
        st.error("Please provide either an anime ID or name for content-based recommendations.")
        return []

    if not idx:
        st.error("Anime not found.")
        return []

    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]

    anime_indices = [i[0] for i in sim_scores]
    recommended_anime = anime_data.iloc[anime_indices]

    return recommended_anime[['name', 'anime_id']].to_dict(orient='records')

def get_collaborative_recommendations(user_id, num_recommendations=10):
    # Prepare the data for Surprise
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[['user_id', 'anime_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    # Predict ratings for all unseen items for the user
    all_anime_ids = anime_data['anime_id'].unique()
    user_anime_ids = ratings[ratings['user_id'] == user_id]['anime_id']
    unseen_anime_ids = [anime_id for anime_id in all_anime_ids if anime_id not in user_anime_ids]
    
    predictions = []
    for anime_id in unseen_anime_ids:
        prediction = svd_model.predict(user_id, anime_id)
        predictions.append((anime_id, prediction.est))
    
    # Sort predictions by estimated rating in descending order
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top N recommendations
    top_recommendations = predictions[:num_recommendations]
    
    # Get the anime details for the top recommendations
    recommended_anime = anime_data[anime_data['anime_id'].isin([rec[0] for rec in top_recommendations])]
    
    return recommended_anime['name'].tolist()

if page == "Recommend Anime":
    st.image(load_image("images/Anime_recommender_logo.jpeg"), width=350)
    st.title("Anime Recommender")
    st.subheader("Discover Your Next Anime Adventure with **AnimeXplore!**")
    st.image(load_image("images/Home_anime_collage.jpg"), use_column_width=True)

    st.markdown("### Choose your recommendation method:")
    rec_method = st.selectbox("Recommendation Method", ["Content-Based Filtering", "Collaborative-Based Filtering"], key="rec_method")

    if rec_method == "Content-Based Filtering":
        st.info("**Tell us what you like, and we'll suggest something you'll love!**")
        search_term = st.text_input("Enter the anime ID or name you like:", key="search_term")

        if st.button("Get Content-Based Recommendations"):
            if search_term:
                recommendations = get_content_based_recommendations(anime_name=search_term)
                if recommendations:
                    st.subheader("Recommended Animes")
                    for anime in recommendations:
                        st.markdown(f"**{anime['name']}**")
                else:
                    st.write("No recommendations found for the given anime.")
            else:
                st.error("Please enter an anime ID or name to get recommendations.")
    
    elif rec_method == "Collaborative-Based Filtering":
        st.subheader("Collaborative-Based Filtering")
        st.info("**Discover Anime Loved by Fans Like You!**")
        user_id = st.text_input("Enter your user ID:", key="user_id")
        
        if st.button("Get Collaborative-Based Recommendations"):
            if user_id:
                try:
                    user_id = int(user_id)
                    recommendations = get_collaborative_recommendations(user_id)
                    if recommendations:
                        st.subheader("Recommended Animes")
                        for anime_name in recommendations:
                            st.markdown(f"**{anime_name}**")
                    else:
                        st.write("No recommendations found for the given user ID.")
                except ValueError:
                    st.error("Invalid user ID. Please enter a numeric user ID.")
            else:
                st.error("Please enter a user ID to get recommendations.")

elif page == "Overview":
    st.title("Welcome to our Anime Recommender App")
    st.info("**Proudly brought to you by AnimeXplore!**")
    st.image(load_image("images/Overview_banner.jpg"), use_column_width=True)

    st.subheader("Your Fun-Filled Anime Quest Begins")
    st.markdown("""
    Imagine a universe where every anime lover finds their perfect match, diving into captivating stories, unforgettable characters, and breathtaking adventures tailored to their unique tastes. At AnimeXplore, we’re on a mission to revolutionize how you discover anime by building a cutting-edge recommender system that’s as vibrant and dynamic as the anime titles it curates. Whether you're a seasoned otaku or a newcomer to the anime world, prepare to embark on an exhilarating journey through the ultimate anime discovery experience.

Anime, a unique form of animation originating from Japan, has a rich history that dates back to the early 20th century. From its humble beginnings with short, silent films, anime has evolved into a global phenomenon, captivating audiences with its diverse genres, intricate plots, and artistic brilliance. Notable milestones in anime history include the release of classics like "Astro Boy" in the 1960s, the rise of Studio Ghibli with timeless masterpieces such as "My Neighbor Totoro" and "Spirited Away," and the explosive popularity of series like "Naruto," "Attack on Titan," and "My Hero Academia."

The impact of anime on global pop culture is undeniable. It has not only entertained millions but also influenced fashion, music, and even technology. Conventions dedicated to anime, such as Anime Expo and Comic-Con, draw massive crowds, celebrating the community and creativity that anime fosters. Streaming platforms now host extensive libraries of anime, making it more accessible than ever before. The stories told through anime resonate deeply with fans, offering both escapism and reflection on real-world issues.
    """)

    st.subheader("Our Objective")
    st.markdown("""
    We aim to develop a collaborative and content-based recommender system that accurately predicts user ratings for unseen anime titles, thereby enhancing the anime discovery experience by delivering personalized, relevant, and exciting recommendations.
    """)
    st.image(load_image("images/anime_fun.gif"), use_column_width=True)

elif page == "Insights":
    st.title("Insights")
    st.info("**Explore Anime Insights and Statistics**")
    
    insights_option = st.selectbox("Choose an insight to view:", 
                                   ["Top 10 Most Rated Animes", "Top 10 Least Rated Animes", "Top 10 Anime Genre Distribution", "Distribution of User Ratings",
                                   "Average Ratings per Genre"], key="insights_option")
    
    if insights_option == "Top 10 Most Rated Animes":
        st.image(load_image("images/top_10_most_rated_animes.png"), use_column_width=True)
    elif insights_option == "Top 10 Least Rated Animes":
        st.image(load_image("images/top_10_least_rated_animes.png"), use_column_width=True)
    elif insights_option == "Top 10 Anime Genre Distribution":
        st.image(load_image("images/top_10_anime_genre_distribution.png"), use_column_width=True)
    elif insights_option == "Distribution of User Ratings":
        st.image(load_image("images/distribution_of_user_ratings.png"), use_column_width=True)
    elif insights_option == "Average Ratings per Genre":
        st.image(load_image("images/Average_Ratings_per_Genre.png"), use_column_width=True)

elif page == "Anime Archive":
    st.title("Anime Archive")
    st.info("**Explore the Vast Anime Collection🔥**")
    st.video("Anime_recommender_video.mp4")
    st.subheader("Search and explore our anime archive.")
    
    search_term = st.text_input("Search for anime by name or genre:", key="archive_search_term")

    if search_term:
        filtered_data = anime_data[anime_data.apply(lambda row: search_term.lower() in row['name'].lower() or search_term.lower() in row['genre'].lower(), axis=1)]
        if not filtered_data.empty:
            for index, row in filtered_data.iterrows():
                st.header(row['name'])
                st.write(f"**Genres:** {row['genre']}")
                st.write(f"**Rating:** {row['rating']}/10")
        else:
            st.write("No anime found matching your search criteria.")

elif page == "About Us":
    st.title("About Us")
    st.subheader("Learn more about this app and its creators.")
    st.image(load_image("images/about_banner.jpg"), use_column_width=True)

    st.markdown("""
    ### About This App:
    This Anime Recommender App is designed to help anime enthusiasts discover new shows to watch based on their preferences. 
    Whether you are a seasoned anime fan or just getting started, this app offers a wide range of features to enhance your anime-watching experience.
    
    ### About the Creators:
    We are a group of passionate anime fans and developers who aim to make anime discovery easier and more enjoyable. Feel free to reach out to us with any feedback or suggestions!
    """)
    st.markdown("### Meet Our Team")
    st.markdown("""
    - **Clement Mphethi** - Lead Data Scientist
    - **Makhutjo Lehutjo** - Project Manager
    - **Prishani Kisten** - Github Manager
    - **Johannes Malefetsane Makgetha** - Data Scientist
    """)
    st.markdown("### Contact Us:")
    st.markdown("For inquiries, please contact us at [info@animexplore.com](mailto:info@animexplore.com).")

st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #637aba;
    text-align: center;
    padding: 10px;
}
</style>
<div class="footer">
    <p>&copy; 2024 Anime Recommender App | Designed with ❤️ by Anime Fans</p>
</div>
""", unsafe_allow_html=True)