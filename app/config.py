import os

# Если контейнер/среда (Railway) уже установил эти переменные, os.getenv их подхватит.
# Если вы локально держите .env, можно вызвать load_dotenv() ДО этих строк,
# чтобы локально .env прочитался и попал в os.environ.

from dotenv import load_dotenv
load_dotenv()  # Откроет файл .env в корне проекта (только локально)

KINOPARK_API_TOKEN = os.getenv("KINOPARK_API_TOKEN", "")
DEEPL_API_KEY      = os.getenv("DEEPL_API_KEY", "")

if not KINOPARK_API_TOKEN:
    raise RuntimeError("Не задана переменная окружения KINOPARK_API_TOKEN")
if not DEEPL_API_KEY:
    raise RuntimeError("Не задана переменная окружения DEEPL_API_KEY")
