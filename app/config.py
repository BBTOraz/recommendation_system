import os
from dotenv import load_dotenv

load_dotenv()


DEEPL_API_KEY = os.getenv('DEEPL_API_KEY')
KINOPARK_API_TOKEN = os.getenv('KINOPARK_API_TOKEN')

print(f"KINOPARK_API_TOKEN: {KINOPARK_API_TOKEN}")
print(f"DEEPL_API_KEY: {DEEPL_API_KEY}")


print(f"KINOPARK_API_TOKEN: ${{shared.DEEPL_API_KEY}}")
print(f"DEEPL_API_KEY: ${{shared.KINOPARK_API_TOKEN}}")

