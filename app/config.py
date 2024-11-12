import os
from dotenv import load_dotenv

load_dotenv()
# Загрузка API-ключей из переменных окружения
DEEPL_API_KEY = os.getenv('DEEPL_API_KEY')
KINOPARK_API_TOKEN = os.getenv('KINOPARK_API_TOKEN')

# Путь к файлу с данными IMDb
IMDB_DATA_PATH = os.getenv(r'C:\Users\Tao\PycharmProjects\recommendation_system_kinopark\data\imdb_movie_data.csv')

