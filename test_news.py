import os, requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GNEWS_API_KEY")

url = "https://gnews.io/api/v4/search"
params = {
    "q": "fraud",
    "lang": "en",
    "country": "in",
    "max": 10,
    "sortby": "publishedAt",
    "apikey": key
}

r = requests.get(url, params=params)
data = r.json()
print("total:", data.get("totalArticles"))
for article in data.get("articles", []):
    print(article["title"], "-", article["publishedAt"])