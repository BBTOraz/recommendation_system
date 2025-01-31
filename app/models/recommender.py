import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.utils.translators import translate_text, translate_and_map_genres
from app.config import KINOPARK_API_TOKEN
import datetime
import logging

class MovieRecommender:
    def __init__(self):
        self.kinopark_df = None
        self.merged_df = None
        self.imdb_data_path = "data/imdb_movie_data.csv"
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.genre_mapping = {
            'cartoon': 'animation',
        }
        self.user_history_normalized = None
        self.load_imdb_data()

    def load_imdb_data(self):
        logging.info("Загрузка данных IMDb")
        self.df = pd.read_csv(self.imdb_data_path)

        self.df = self.df[self.df['Year'] >= 2012]

        self.df = self.df.sample(n=5000, random_state=42)

        self.df['source'] = 'imdb'
        text_columns = ['genres', 'directors', 'writers', 'actors']
        for col in text_columns:
            self.df[col] = self.df[col].fillna('')
        self.df['Movie'] = self.df['Movie'].fillna('Unknown')
        self.df['Year'] = self.df['Year'].fillna('Unknown')
        self.df['content'] = (
            self.df['genres'] + ' ' +
            self.df['directors'] + ' ' +
            self.df['writers'] + ' ' +
            self.df['actors']
        )

    def get_kinopark_data(self, city_id):
        logging.info("Получение текущей даты в формате YYYY-MM-DD")
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
                logging.warning("Нет данных о фильмах для указанного города.")
                self.kinopark_df = pd.DataFrame()
        else:
            logging.error(f"Ошибка при получении данных из Kinopark API для города {city_id}")
            self.kinopark_df = pd.DataFrame()

    def translate_text_no_cache(self, text):
        if not isinstance(text, str):
            text = str(text)
        return translate_text(text)

    def translate_and_map_genres_no_cache(self, genres_tuple):
        return translate_and_map_genres(genres_tuple, self.genre_mapping)

    def process_kinopark_data(self):
        logging.info("Перевод и обработка данных Kinopark")
        if not self.kinopark_df.empty:
            self.kinopark_df['Year'] = pd.to_datetime(self.kinopark_df['release_date']).dt.year.astype(str)
            self.kinopark_df['name_en'] = self.kinopark_df['name'].apply(lambda x: self.translate_text_no_cache(x))
            self.kinopark_df['genre_str'] = self.kinopark_df['genre'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else ''
            )
            self.kinopark_df['trailer'] = self.kinopark_df['trailer.url']
            self.kinopark_df['imageVertical'] = self.kinopark_df['image.vertical']
            self.kinopark_df['imageHorizontal'] = self.kinopark_df['image.horizontal']
            self.kinopark_df['seanceTimeframes'] = self.kinopark_df['seance.timeframes']
            self.kinopark_df['genre_en'] = self.kinopark_df['genre'].apply(
                lambda x: self.translate_and_map_genres_no_cache(tuple(x)) if isinstance(x, list) else ''
            )
            self.kinopark_df['directors_str'] = self.kinopark_df['directors'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else ''
            )
            self.kinopark_df['directors_en'] = self.kinopark_df['directors'].apply(
                lambda x: ', '.join([self.translate_text_no_cache(director) for director in x]) if isinstance(x, list) else ''
            )
            self.kinopark_df['actors_str'] = self.kinopark_df['actors'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else ''
            )
            self.kinopark_df['actors_en'] = self.kinopark_df['actors'].apply(
                lambda x: ', '.join([self.translate_text_no_cache(actor) for actor in x]) if isinstance(x, list) else ''
            )
            self.kinopark_df['Movie'] = self.kinopark_df['name_en'] + ' (' + self.kinopark_df['Year'] + ')'
            self.kinopark_df['content'] = (
                self.kinopark_df['genre_en'] + ' ' +
                self.kinopark_df['directors_en'] + ' ' +
                self.kinopark_df['actors_en']
            )
            self.kinopark_df['source'] = 'kinopark'
        else:
            logging.warning("Kinopark DataFrame is empty.")

    def merge_datasets(self):
        logging.info("Объединение IMDb и Kinopark данных")
        necessary_columns = [
            'Movie', 'Year', 'content', 'source', 'id', 'name', 'trailer',
            'imageVertical', 'imageHorizontal', 'similarity'
        ]
        logging.info("Приведение столбцов к общему виду")
        for col in necessary_columns:
            if col not in self.df.columns:
                self.df[col] = None
            if col not in self.kinopark_df.columns:
                self.kinopark_df[col] = None
        self.merged_df = pd.concat([self.df, self.kinopark_df], ignore_index=True, sort=False)

    def prepare_tfidf(self):
        logging.info("Обучение TF-IDF модели на объединённых данных")

        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=2000,
            max_df=0.8,
            min_df=5
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.merged_df['content'].fillna(''))

    def prepare_kinopark_tfidf(self):
        logging.info("Преобразование контента Kinopark с помощью TF-IDF")
        if not self.kinopark_df.empty:
            self.kinopark_tfidf = self.tfidf_vectorizer.transform(self.kinopark_df['content'].fillna(''))
        else:
            self.kinopark_tfidf = None

    def create_user_profile(self, user_history):
        logging.info("Нормализация названий фильмов из истории пользователя")
        self.user_history_normalized = [self.translate_text_no_cache(title) for title in user_history]
        user_indices = self.merged_df[self.merged_df['Movie'].isin(self.user_history_normalized)].index
        if not user_indices.empty:
            user_tfidf = self.tfidf_matrix[user_indices]
            user_profile = user_tfidf.mean(axis=0)
            user_profile = np.asarray(user_profile)
            return user_profile
        else:
            logging.warning("Фильмы из истории пользователя не найдены в объединенном датасете.")
            return None

    def recommended_movies(self, user_profile, top_n=10):
        logging.info("Рекомендация фильмов пользователю")
        if user_profile is not None and self.kinopark_tfidf is not None:
            similarities = cosine_similarity(user_profile, self.kinopark_tfidf).flatten()
            recommendations = self.kinopark_df.copy()
            recommendations['similarity'] = similarities
            recommendations = recommendations[~recommendations['Movie'].isin(self.user_history_normalized)]
            top_recommendations = recommendations.sort_values(by='similarity', ascending=False).head(top_n)
            logging.info("Рекомендация фильмов пользователю завершена")
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
                logging.warning(f"Отсутствуют следующие столбцы: {missing_columns}")
                for col in missing_columns:
                    top_recommendations[col] = None
            # Возвращаем только нужные столбцы
            return top_recommendations[result_columns]
        else:
            logging.warning("Не удалось создать профиль пользователя или нет данных Kinopark.")
            return None
