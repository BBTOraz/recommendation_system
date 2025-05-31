import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.utils.translators import translate_text, translate_and_map_genres
from app.config import KINOPARK_API_TOKEN
import datetime
import logging
from typing import List

logging.basicConfig(level=logging.INFO)

class MovieRecommender:
    def __init__(self):
        # DataFrames
        self.kinopark_df = pd.DataFrame()
        self.df = None               # IMDb DataFrame
        self.merged_df = None        # объединённый DataFrame

        # Пути и модели TF-IDF
        self.imdb_data_path = "data/imdb_movie_data.csv"
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.kinopark_tfidf = None

        # Маппинг для перевода жанров
        self.genre_mapping = {
            'cartoon': 'animation',
        }

        # Для хранения «англоязычных» Movie из истории пользователя
        self.user_history_normalized = None

        # Сразу подгружаем IMDb
        self.load_imdb_data()

    def load_imdb_data(self):
        """
        Загружаем CSV с IMDb и формируем поле 'content' = genres только.
        Предполагается, что CSV уже на английском:
          - 'Movie' — английский заголовок,
          - 'Year' — год,
          - 'genres' — строка вида "Action, Comedy, Drama".
        Мы не используем actors/directors/writers.
        """
        logging.info("Загрузка данных IMDb")
        self.df = pd.read_csv(self.imdb_data_path)

        # Оставляем фильмы с 2012 и позже
        self.df = self.df[self.df['Year'] >= 2012]
        self.df = self.df.sample(n=5000, random_state=42)

        # Источник
        self.df['source'] = 'imdb'

        # Заполним NaN
        if 'genres' not in self.df.columns:
            self.df['genres'] = ''
        else:
            self.df['genres'] = self.df['genres'].fillna('')

        self.df['Movie'] = self.df['Movie'].fillna('Unknown')
        self.df['Year'] = self.df['Year'].fillna('Unknown').astype(str)

        # Формируем поле content (только жанры)
        self.df['content'] = self.df['genres'].astype(str)

        # Movie в формате "English Title (Year)"
        self.df['Movie'] = self.df['Movie'].astype(str) + ' (' + self.df['Year'].astype(str) + ')'

    def get_kinopark_data(self, city_id: str):
        """
        Загружает JSON из Kinopark API и сохраняет в self.kinopark_df.
        """
        logging.info("Получение данных Kinopark API")
        current_date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        url = f"https://afisha.api.kinopark.kz/api/movie/today?city={city_id}&start={current_date}"
        headers = {
            "Authorization": f"Bearer {KINOPARK_API_TOKEN}",
            "Accept": "application/json"
        }
        try:
            response = requests.get(url, headers=headers)
        except Exception as e:
            logging.error(f"Ошибка при запросе к Kinopark API: {e}")
            self.kinopark_df = pd.DataFrame()
            return

        if response.status_code == 200:
            kinopark_data = response.json()
            if 'data' in kinopark_data and kinopark_data['data']:
                self.kinopark_df = pd.json_normalize(kinopark_data['data'])
                logging.info(f"Получено {len(self.kinopark_df)} фильмов из Kinopark")
            else:
                logging.warning("Нет данных о фильмах для указанного города.")
                self.kinopark_df = pd.DataFrame()
        else:
            logging.error(f"Ошибка Kinopark API, код {response.status_code}")
            self.kinopark_df = pd.DataFrame()

    def translate_text_safe(self, text: str) -> str:
        """
        Оболочка для translate_text, чтобы не падать в случае ошибок API.
        При ошибке возвращаем исходный текст (русский), чтобы Movie не выпал из поиска.
        """
        try:
            return translate_text(text)
        except Exception as e:
            logging.error(f"Ошибка при переводе текста '{text}': {e}")
            return text  # возвращаем исходное значение

    def process_kinopark_data(self):
        """
        Обрабатываем DataFrame Kinopark:
        - Переводим только поле 'name' → 'name_en'
        - Преобразуем genre → 'genre_list' (английские жанры) и 'genre_str'
        - Дублируем жанры в 'genre_str_weighted'
        - Инициализируем trailer, imageVertical, imageHorizontal
        - Собираем 'Movie' = "{name_en} (Year)"
        - Собираем 'content' = name_en + genre_str_weighted
        """
        logging.info("Обработка данных Kinopark (без описания, без актёров/режиссёров)")
        if self.kinopark_df.empty:
            logging.warning("Kinopark DataFrame пуст.")
            return

        # 1) Год
        self.kinopark_df['Year'] = pd.to_datetime(
            self.kinopark_df.get('release_date', None),
            errors='coerce'
        ).dt.year.fillna('Unknown').astype(str)

        # 2) Перевод названия → name_en
        self.kinopark_df['name_en'] = self.kinopark_df['name'].apply(
            lambda x: self.translate_text_safe(str(x))
        )

        # 3) Преобразуем genre → genre_list (английские) и genre_str
        if 'genre' in self.kinopark_df.columns:
            # translate_and_map_genres возвращает строку "action, comedy"
            self.kinopark_df['genre_list'] = self.kinopark_df['genre'].apply(
                lambda x: translate_and_map_genres(tuple(x), self.genre_mapping).split(', ')
                if isinstance(x, list) else []
            )
        else:
            self.kinopark_df['genre_list'] = []

        self.kinopark_df['genre_str'] = self.kinopark_df['genre_list'].apply(
            lambda lst: ', '.join(lst)
        )

        # 4) Дублируем жанры 3 раза для усиления веса
        self.kinopark_df['genre_str_weighted'] = self.kinopark_df['genre_str'].apply(
            lambda s: (s + ' ') * 3 if s else ''
        )

        # 5) Трейлеры и картинки (прямо из JSON)
        self.kinopark_df['trailer'] = self.kinopark_df.get('trailer.url', None)
        self.kinopark_df['imageVertical'] = self.kinopark_df.get('image.vertical', None)
        self.kinopark_df['imageHorizontal'] = self.kinopark_df.get('image.horizontal', None)

        # 6) Movie = "{name_en} (Year)"
        self.kinopark_df['Movie'] = (
            self.kinopark_df['name_en'].fillna('').astype(str) +
            ' (' + self.kinopark_df['Year'].astype(str).fillna('') + ')'
        )

        # 7) content = name_en + genre_str_weighted
        self.kinopark_df['content'] = (
            self.kinopark_df['name_en'].fillna('').astype(str) + ' ' +
            self.kinopark_df['genre_str_weighted'].fillna('').astype(str)
        )

        # 8) Источник
        self.kinopark_df['source'] = 'kinopark'

        logging.info("Kinopark DataFrame после обработки:")
        logging.info(self.kinopark_df[['name', 'name_en', 'genre_str', 'genre_str_weighted', 'trailer', 'imageVertical', 'imageHorizontal', 'content']].head())

    def merge_datasets(self):
        """
        Объединяем IMDb и Kinopark в merged_df.
        Если Kinopark пуст, просто копируем IMDb.
        """
        logging.info("Объединение IMDb и Kinopark")
        if self.kinopark_df is None or self.kinopark_df.empty:
            self.merged_df = self.df.copy()
            return

        necessary_columns = [
            'Movie', 'Year', 'content', 'source', 'id', 'name',
            'genre_list', 'genre_str', 'genre_str_weighted',
            'trailer', 'imageVertical', 'imageHorizontal'
        ]
        for col in necessary_columns:
            if col not in self.df.columns:
                self.df[col] = None
            if col not in self.kinopark_df.columns:
                self.kinopark_df[col] = None

        self.merged_df = pd.concat([self.df, self.kinopark_df], ignore_index=True, sort=False)
        self.merged_df['content'] = self.merged_df['content'].fillna('').astype(str)
        self.merged_df['Movie'] = self.merged_df['Movie'].fillna('').astype(str)

    def prepare_tfidf(self):
        """
        Обучаем TF-IDF на объединённом поле 'content'.
        """
        logging.info("Обучение TF-IDF на merged_df")
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words=None,
            max_features=2000,
            max_df=0.8,
            min_df=5
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.merged_df['content'])

    def prepare_kinopark_tfidf(self):
        """
        Строим TF-IDF-матрицу только для Kinopark-фильмов.
        """
        logging.info("Построение TF-IDF для Kinopark")
        kinopark_mask = self.merged_df['source'] == 'kinopark'
        kinopark_indices = np.where(kinopark_mask)[0]
        if len(kinopark_indices) > 0:
            self.kinopark_tfidf = self.tfidf_matrix[kinopark_indices]
        else:
            self.kinopark_tfidf = None

    def create_user_profile(self, user_history: List[str]):
        """
        user_history — список строк "English Title (Year)".
        Возвращает усреднённый TF-IDF-вектор (1×features) или None, если ничего не найдено.
        """
        logging.info("Создание профиля пользователя")
        self.user_history_normalized = [title.strip() for title in user_history]

        mask = self.merged_df['Movie'].isin(self.user_history_normalized)
        user_indices = np.where(mask)[0]
        if len(user_indices) == 0:
            logging.warning("Ни один фильм из истории пользователя не найден.")
            return None

        user_tfidf = self.tfidf_matrix[user_indices]
        user_profile = user_tfidf.mean(axis=0)
        return np.asarray(user_profile)

    def recommended_movies(self, user_profile, top_n: int = 10):
        """
        Собираем список топ-N фильмов из Kinopark по cosine_similarity.
        Возвращаем поля ['id','name','trailer','imageVertical','imageHorizontal','similarity'].
        """
        logging.info("Рекомендация фильмов по профилю пользователя")
        if user_profile is None:
            logging.warning("Профиль пользователя не создан.")
            return None
        if self.kinopark_tfidf is None:
            logging.warning("TF-IDF для Kinopark не готов.")
            return None

        similarities = cosine_similarity(user_profile, self.kinopark_tfidf).flatten()

        kinopark_mask = self.merged_df['source'] == 'kinopark'
        kinopark_df_only = self.merged_df[kinopark_mask].copy().reset_index(drop=True)
        kinopark_df_only['similarity'] = similarities

        # Исключаем те, что уже есть в истории
        kinopark_df_only = kinopark_df_only[~kinopark_df_only['Movie'].isin(self.user_history_normalized)]

        # Сортировка и топ-N
        top_recs = kinopark_df_only.sort_values(by='similarity', ascending=False).head(top_n)

        result_columns = [
            'id',
            'name',
            'trailer',
            'imageVertical',
            'imageHorizontal',
            'similarity'
        ]
        for col in result_columns:
            if col not in top_recs.columns:
                top_recs[col] = None

        return top_recs[result_columns]

    def recommended_by_translated_title(self, rus_movie_title: str, top_n: int = 5):
        """
        Вход: "Русский Заголовок (Год)".
        Переводим rus_title → eng_title. Если перевод не удался, eng_title = rus_title.
        Формируем "eng_title (Год)" и ищем в merged_df['Movie'].
        Если нашли, считаем cosine_similarity TF-IDF с Kinopark и возвращаем топ-N.
        Итоговые поля: ['id','name','trailer','imageVertical','imageHorizontal','similarity'].
        """
        logging.info(f"Поиск рекомендаций по русскому заголовку: {rus_movie_title}")
        movie_str = rus_movie_title.strip()
        if "(" in movie_str and movie_str.endswith(")"):
            idx_paren = movie_str.rfind("(")
            rus_title = movie_str[:idx_paren].strip()
            year = movie_str[idx_paren+1:-1].strip()
        else:
            logging.warning("Неправильный формат 'Название (Год)'.")
            return None

        eng_title = self.translate_text_safe(rus_title)
        translated_movie = f"{eng_title} ({year})"
        logging.info(f"Переведённое название: {translated_movie}")

        mask = self.merged_df['Movie'] == translated_movie
        indices = np.where(mask)[0]
        if len(indices) == 0:
            logging.warning(f"Фильм '{translated_movie}' не найден.")
            return None
        idx = indices[0]

        movie_vector = self.tfidf_matrix[idx]
        if self.kinopark_tfidf is None:
            logging.warning("TF-IDF для Kinopark не готов.")
            return None

        similarities = cosine_similarity(movie_vector, self.kinopark_tfidf).flatten()

        kinopark_mask = self.merged_df['source'] == 'kinopark'
        kinopark_df_only = self.merged_df[kinopark_mask].copy().reset_index(drop=True)
        kinopark_df_only['similarity'] = similarities

        # Исключаем сам фильм
        kinopark_df_only = kinopark_df_only[kinopark_df_only['Movie'] != translated_movie]

        top_recs = kinopark_df_only.sort_values(by='similarity', ascending=False).head(top_n)

        result_columns = [
            'id',
            'name',
            'trailer',
            'imageVertical',
            'imageHorizontal',
            'similarity'
        ]
        for col in result_columns:
            if col not in top_recs.columns:
                top_recs[col] = None

        return top_recs[result_columns]
