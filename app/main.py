import os

from fastapi import FastAPI
from app.routers import recommendations

app = FastAPI(
    title="Movie Recommendation API",
    description="API для предоставления рекомендаций фильмов",
    version="1.0.0",
)
app.include_router(recommendations.router)

@app.get("/")
def read_root():
    return {"message": "Добро пожаловать в Movie Recommendation API"}

KINOPARK_API_TOKEN = os.getenv("KINOPARK_API_TOKEN")
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")

@app.get("/check-env")
def check_env():
    return {
        "KINOPARK_API_TOKEN": KINOPARK_API_TOKEN,
        "DEEPL_API_KEY": DEEPL_API_KEY
    }