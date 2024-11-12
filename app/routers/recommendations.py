# app/routers/recommendations.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.models.recommender import MovieRecommender

router = APIRouter(
    prefix="/recommendations",
    tags=["recommendations"]
)

recommender = MovieRecommender()

class RecommendationRequest(BaseModel):
    user_history: List[str]
    city_id: str
    top_n: int = 10

@router.post("/")
def get_recommendations(request: RecommendationRequest):
    print("get_recommendations method")
    recommender.get_kinopark_data(request.city_id)
    recommender.process_kinopark_data()
    recommender.prepare_kinopark_tfidf()
    user_profile = recommender.create_user_profile(request.user_history)
    recommendations = recommender.recommended_movies(user_profile, top_n=request.top_n)
    if recommendations is not None:
        return recommendations.to_dict(orient='records')
    else:
        raise HTTPException(status_code=404, detail="Не удалось получить рекомендации")
