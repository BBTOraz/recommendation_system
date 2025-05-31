from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.models.recommender import MovieRecommender

router = APIRouter(
    prefix="/recommendations",
    tags=["recommendations"]
)

# Один раз инициализируем MovieRecommender (он сразу подгружает IMDb)
recommender = MovieRecommender()

class RecommendationRequest(BaseModel):
    user_history: List[str]      # Список строк вида "Русский Заголовок (Год)"
    city_id: str                 # Id города для Kinopark API
    top_n: int = 10              # Сколько рекомендаций вернуть

@router.post("/")
def get_recommendations(request: RecommendationRequest):
    """
    1. Загружаем/обновляем Kinopark-данные.
    2. Если user_history содержит ровно 1 фильм (русский заголовок), вызываем recommended_by_translated_title.
    3. Если >1, переводим каждую строку в "English Title (Year)", формируем профиль и вызываем recommended_movies.
    """
    try:
        # Шаг 1: загрузка/обновление Kinopark
        recommender.get_kinopark_data(request.city_id)
        recommender.process_kinopark_data()
        recommender.merge_datasets()
        recommender.prepare_tfidf()
        recommender.prepare_kinopark_tfidf()

        # Шаг 2: если ровно 1 фильм, считаем, что это «Русский Заголовок (Год)»
        if len(request.user_history) == 1:
            rus_movie = request.user_history[0].strip()
            recommendations = recommender.recommended_by_translated_title(
                rus_movie_title=rus_movie,
                top_n=request.top_n
            )
            if recommendations is None or recommendations.empty:
                raise HTTPException(status_code=404, detail="Фильм не найден или рекомендаций нет")
            return recommendations.to_dict(orient="records")

        # Шаг 3: если >1 фильма, переводим каждое название и строим профиль
        english_history: List[str] = []
        for rus_movie_str in request.user_history:
            m = rus_movie_str.strip()
            if "(" in m and m.endswith(")"):
                idx = m.rfind("(")
                rus_title = m[:idx].strip()
                year = m[idx+1:-1].strip()
            else:
                raise HTTPException(status_code=400, detail=f"Неправильный формат '{m}'. Ожидается: 'Название (Год)'")

            eng_title = recommender.translate_text_safe(rus_title)
            translated_full = f"{eng_title} ({year})"
            english_history.append(translated_full)

        user_profile = recommender.create_user_profile(english_history)
        if user_profile is None:
            raise HTTPException(status_code=404, detail="Ни один фильм из истории не найден")

        recommendations = recommender.recommended_movies(user_profile, top_n=request.top_n)
        if recommendations is None or recommendations.empty:
            raise HTTPException(status_code=404, detail="Не удалось получить рекомендации")
        return recommendations.to_dict(orient="records")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Ошибка при получении рекомендаций: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")
