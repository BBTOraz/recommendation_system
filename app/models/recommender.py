# app/models/recommender.py

import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.utils.translators import translate_text, translate_and_map_genres
from app.config import KINOPARK_API_TOKEN

class MovieRecommender:
    def __init__(self):
        self.imdb_data_path = r"C:\Users\Tao\PycharmProjects\recommendation_system_kinopark\data\imdb_movie_data.csv"
        self.df = None  # IMDb DataFrame
        self.kinopark_df = None  # Kinopark DataFrame
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None  # TF-IDF матрица для IMDb
        self.kinopark_tfidf = None  # TF-IDF матрица для Kinopark
        self.genre_mapping = {
            'cartoon': 'animation',
            # Добавьте другие соответствия по необходимости
        }
        self.load_imdb_data()
        self.prepare_tfidf()

    def load_imdb_data(self):
        # Загрузка и обработка данных IMDb
        self.df = pd.read_csv(self.imdb_data_path)
        text_columns = ['genres', 'directors', 'writers', 'actors']
        for col in text_columns:
            self.df[col] = self.df[col].fillna('')
        self.df['Movie'] = self.df['Movie'].fillna('Unknown')
        self.df['Year'] = self.df['Year'].fillna('Unknown')
        self.df['content'] = self.df['genres'] + ' ' + self.df['directors'] + ' ' + self.df['writers'] + ' ' + self.df['actors']

    def get_kinopark_data(self, city_id):
        # Получение данных из Kinopark API
        url = f"https://afisha.api.kinopark.kz/api/movie/today?city={city_id}&start=2024-11-12"
        headers = {
            "Authorization": f"Bearer {KINOPARK_API_TOKEN}",
            "Accept": "application/json"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            kinopark_data = response.json()
            self.kinopark_df = pd.json_normalize(kinopark_data['data'])
            print(self.kinopark_df.columns)
            print(self.kinopark_df[["image.vertical", "trailer.url", "image.horizontal"]])
        else:
            print("Ошибка при получении данных из Kinopark API")
            self.kinopark_df = pd.DataFrame()

    def process_kinopark_data(self):
        # Перевод и обработка данных Kinopark
        if not self.kinopark_df.empty:
            self.kinopark_df['Year'] = pd.to_datetime(self.kinopark_df['release_date']).dt.year.astype(str)
            self.kinopark_df['name_en'] = self.kinopark_df['name'].apply(lambda x: translate_text(x))
            self.kinopark_df['genre_str'] = self.kinopark_df['genre'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else ''
            )
            self.kinopark_df['trailer'] = self.kinopark_df['trailer.url']
            self.kinopark_df['imageVertical'] = self.kinopark_df['image.vertical']
            self.kinopark_df['imageHorizontal'] = self.kinopark_df['image.horizontal']
            self.kinopark_df['seanceTimeframes'] = self.kinopark_df['seance.timeframes']
            self.kinopark_df['genre_en'] = self.kinopark_df['genre'].apply(
                lambda x: translate_and_map_genres(x, self.genre_mapping) if isinstance(x, list) else ''
            )
            self.kinopark_df['directors_str'] = self.kinopark_df['directors'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else ''
            )
            self.kinopark_df['directors_en'] = self.kinopark_df['directors'].apply(
                lambda x: ', '.join([translate_text(director) for director in x]) if isinstance(x, list) else ''
            )

            self.kinopark_df['actors_str'] = self.kinopark_df['actors'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else ''
            )
            self.kinopark_df['actors_en'] = self.kinopark_df['actors'].apply(
                lambda x: ', '.join([translate_text(actor) for actor in x]) if isinstance(x, list) else ''
            )
            self.kinopark_df['Movie'] = self.kinopark_df['name_en'] + ' (' + self.kinopark_df['Year'] + ')'
            self.kinopark_df['content'] = (
                    self.kinopark_df['genre_en'] + ' ' +
                    self.kinopark_df['directors_en'] + ' ' +
                    self.kinopark_df['actors_en']
            )
            print(self.kinopark_df.columns)
        else:
            print("Kinopark DataFrame is empty.")

    def prepare_tfidf(self):
        # Обучение TF-IDF модели на данных IMDb
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['content'])

    def prepare_kinopark_tfidf(self):
        if not self.kinopark_df.empty:
            self.kinopark_tfidf = self.tfidf_vectorizer.transform(self.kinopark_df['content'])
        else:
            print("Kinopark DataFrame is empty.")

    def create_user_profile(self, user_history):
        user_indices = self.df[self.df['Movie'].isin(user_history)].index
        if not user_indices.empty:
            user_tfidf = self.tfidf_matrix[user_indices]
            user_profile = user_tfidf.mean(axis=0)
            user_profile = np.asarray(user_profile)
            return user_profile
        else:
            print("Фильмы из истории пользователя не найдены в датасете.")
            return None

    def recommended_movies(self, user_profile, top_n=10):
        # Генерация рекомендаций на основе профиля пользователя
        if user_profile is not None and self.kinopark_tfidf is not None:
            similarities = cosine_similarity(user_profile, self.kinopark_tfidf).flatten()
            recommendations = self.kinopark_df.copy()
            recommendations['similarity'] = similarities
            top_recommendations = recommendations.sort_values(by='similarity', ascending=False).head(top_n)
            result_columns = [
                'id',
                'name',
                'trailer',
                'imageVertical',
                'imageHorizontal',
                'similarity'
            ]
            # Проверяем, что все необходимые столбцы существуют
            missing_columns = [col for col in result_columns if col not in top_recommendations.columns]
            if missing_columns:
                print(f"Отсутствуют следующие столбцы: {missing_columns}")
                # Вы можете заполнить отсутствующие столбцы значениями по умолчанию или обработать иначе

            # Возвращаем только нужные столбцы
            return top_recommendations[result_columns]
        else:
            print("Не удалось создать профиль пользователя или TF-IDF для Kinopark.")
            return None
