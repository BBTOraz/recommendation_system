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
