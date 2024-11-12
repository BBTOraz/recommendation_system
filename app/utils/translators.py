import requests
from app.config import DEEPL_API_KEY

def translate_text(text, target_language='EN'):
    api_url = 'https://api-free.deepl.com/v2/translate'
    params = {
        'auth_key': DEEPL_API_KEY,
        'text': text,
        'target_lang': target_language
    }
    try:
        response = requests.post(api_url, data=params)
        response.raise_for_status()
        result = response.json()
        return result['translations'][0]['text']
    except Exception as e:
        print(f"Ошибка при переводе текста: {e}")
        return text

def translate_and_map_genres(genres_list, genre_mapping):
    mapped_genres = []
    for genre_entry in genres_list:
        individual_genres = genre_entry.split(',')
        for genre in individual_genres:
            genre = genre.strip()
            translated = translate_text(genre)
            formatted_genre = translated.strip().lower()
            mapped_genre = genre_mapping.get(formatted_genre, formatted_genre)
            mapped_genres.append(mapped_genre.title())
    return ' '.join(mapped_genres)
