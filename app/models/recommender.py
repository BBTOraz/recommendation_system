# app/models/recommender.py

import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.utils.translators import translate_text, translate_and_map_genres
from app.config import KINOPARK_API_TOKEN
from functools import lru_cache
import datetime
import threading

class MovieRecommender:
    def __init__(self):
        self.imdb_data_path = "data/imdb_movie_data.csv"
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.genre_mapping = {
            'cartoon': 'animation',
        }
        self.load_imdb_data()
        self.user_history_normalized = None


    def load_imdb_data(self):
        print("Загрузка данных IMDb")
        self.df = pd.read_csv(self.imdb_data_path)
        self.df['source'] = 'imdb'
        text_columns = ['genres', 'directors', 'writers', 'actors']
        for col in text_columns:
            self.df[col] = self.df[col].fillna('')
        self.df['Movie'] = self.df['Movie'].fillna('Unknown')
        self.df['Year'] = self.df['Year'].fillna('Unknown')
        self.df['content'] = self.df['genres'] + ' ' + self.df['directors'] + ' ' + self.df['writers'] + ' ' + self.df['actors']

    def get_kinopark_data(self, city_id):
        print("Получение текущей даты в формате YYYY-MM-DD")
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        url = f"https://afisha.api.kinopark.kz/api/movie/today?city={city_id}&start={current_date}"
        headers = {
            "Authorization": f"Bearer {KINOPARK_API_TOKEN}",
            "Accept": "application/json"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            kinopark_data = response.json()
            if 'data' in kinopark_data and kinopark_data['data']:
                self.kinopark_df = pd.json_normalize(kinopark_data['data'])
            else:
                print("Нет данных о фильмах для указанного города.")
                self.kinopark_df = pd.DataFrame()
        else:
            print(f"Ошибка при получении данных из Kinopark API для города {city_id}")
            self.kinopark_df = pd.DataFrame()

    @lru_cache(maxsize=10000)
    def cached_translate_text(self, text):
        print("кеширование перевода текста")
        if not isinstance(text, str):
            text = str(text)
        return translate_text(text)

    @lru_cache(maxsize=10000)
    def cached_translate_and_map_genres(self, genres_tuple):
        print("кеширование перевод и маппинг текста")
        return translate_and_map_genres(genres_tuple, self.genre_mapping)

    def process_kinopark_data(self):
        print("Перевод и обработка данных Kinopark")
        if not self.kinopark_df.empty:
            self.kinopark_df['Year'] = pd.to_datetime(self.kinopark_df['release_date']).dt.year.astype(str)
            self.kinopark_df['name_en'] = self.kinopark_df['name'].apply(lambda x: self.cached_translate_text(x))
            self.kinopark_df['genre_str'] = self.kinopark_df['genre'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else ''
            )
            self.kinopark_df['trailer'] = self.kinopark_df['trailer.url']
            self.kinopark_df['imageVertical'] = self.kinopark_df['image.vertical']
            self.kinopark_df['imageHorizontal'] = self.kinopark_df['image.horizontal']
            self.kinopark_df['seanceTimeframes'] = self.kinopark_df['seance.timeframes']
            self.kinopark_df['genre_en'] = self.kinopark_df['genre'].apply(
                lambda x: self.cached_translate_and_map_genres(tuple(x)) if isinstance(x, list) else ''
            )
            self.kinopark_df['directors_str'] = self.kinopark_df['directors'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else ''
            )
            self.kinopark_df['directors_en'] = self.kinopark_df['directors'].apply(
                lambda x: ', '.join([self.cached_translate_text(director) for director in x]) if isinstance(x,
                                                                                                            list) else ''
            )
            self.kinopark_df['actors_str'] = self.kinopark_df['actors'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else ''
            )
            self.kinopark_df['actors_en'] = self.kinopark_df['actors'].apply(
                lambda x: ', '.join([self.cached_translate_text(actor) for actor in x]) if isinstance(x, list) else ''
            )
            self.kinopark_df['Movie'] = self.kinopark_df['name_en'] + ' (' + self.kinopark_df['Year'] + ')'
            self.kinopark_df['content'] = (
                    self.kinopark_df['genre_en'] + ' ' +
                    self.kinopark_df['directors_en'] + ' ' +
                    self.kinopark_df['actors_en']
            )
            self.kinopark_df['source'] = 'kinopark'
        else:
            print("Kinopark DataFrame is empty.")

    def merge_datasets(self):
        print("Объединение IMDb и Kinopark данных")
        necessary_columns = [
            'Movie', 'Year', 'content', 'source', 'id', 'name', 'trailer',
            'imageVertical', 'imageHorizontal', 'similarity'
        ]
        print("Приведение столбцов к общему виду")
        for col in necessary_columns:
            if col not in self.df.columns:
                self.df[col] = None
            if col not in self.kinopark_df.columns:
                self.kinopark_df[col] = None
        self.merged_df = pd.concat([self.df, self.kinopark_df], ignore_index=True, sort=False)

    def prepare_tfidf(self):
        print("Обучение TF-IDF модели на объединённых данных")
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.merged_df['content'].fillna(''))

    def prepare_kinopark_tfidf(self):
        print("Преобразование контента Kinopark с помощью TF-IDF")
        if not self.kinopark_df.empty:
            self.kinopark_tfidf = self.tfidf_vectorizer.transform(self.kinopark_df['content'].fillna(''))
        else:
            self.kinopark_tfidf = None

    def create_user_profile(self, user_history):
        print("Нормализация названий фильмов из истории пользователя")
        self.user_history_normalized = [self.cached_translate_text(title) for title in user_history]
        """print("Нормализованные названия из истории пользователя:", self.user_history_normalized)
        print("Доступные фильмы:", self.merged_df['Movie'].tolist())"""
        user_indices = self.merged_df[self.merged_df['Movie'].isin(self.user_history_normalized)].index
        if not user_indices.empty:
            user_tfidf = self.tfidf_matrix[user_indices]
            user_profile = user_tfidf.mean(axis=0)
            user_profile = np.asarray(user_profile)
            return user_profile
        else:
            print("Фильмы из истории пользователя не найдены в объединенном датасете.")
            return None

    def recommended_movies(self, user_profile, top_n=10):
        print("Рекомендация фильмов пользователю")
        if user_profile is not None and self.kinopark_tfidf is not None:
            similarities = cosine_similarity(user_profile, self.kinopark_tfidf).flatten()
            recommendations = self.kinopark_df.copy()
            recommendations['similarity'] = similarities
            recommendations = recommendations[~recommendations['Movie'].isin(self.user_history_normalized)]
            top_recommendations = recommendations.sort_values(by='similarity', ascending=False).head(top_n)
            print("Рекомендация фильмов пользователю завершена")
            result_columns = [
                'id',
                'name',
                'trailer',
                'imageVertical',
                'imageHorizontal',
                'similarity'
            ]
            missing_columns = [col for col in result_columns if col not in top_recommendations.columns]
            if missing_columns:
                print(f"Отсутствуют следующие столбцы: {missing_columns}")
                for col in missing_columns:
                    top_recommendations[col] = None
            # Возвращаем только нужные столбцы
            return top_recommendations[result_columns]
        else:
            print("Не удалось создать профиль пользователя или нет данных Kinopark.")
            return None
