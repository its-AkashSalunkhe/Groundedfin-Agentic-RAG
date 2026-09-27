'''

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

'''
'''

import requests, re
from pypdf import PdfReader

PRIDS = [43098, 44392, 44185, 32572, 32908, 33674, 31276, 27218, 25911, 27405]

def clean(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fetch_press_release(prid):
    url = f"https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid={prid}"
    r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    text = r.text
    m = re.search(r'Date\s*:\s*.*?\|(.*?)Related Press Releases', text, re.S)
    body = clean(m.group(1)) if m else clean(text)
    return body

if __name__ == "__main__":
    import os
    os.makedirs("data/rbi_docs/press_releases", exist_ok=True)
    for prid in PRIDS:
        try:
            body = fetch_press_release(prid)
            with open(f"data/rbi_docs/press_releases/pr_{prid}.txt", "w", encoding="utf-8") as f:
                f.write(body)
            print(prid, "ok,", len(body), "chars")
        except Exception as e:
            print(prid, "FAILED:", e)

'''



import requests, re
from bs4 import BeautifulSoup

PRIDS = [43098, 44392, 44185, 32572, 32908, 33674, 31276, 27218, 25911, 27405]

def fetch_press_release(prid):
    url = f"https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid={prid}"
    r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r'\n\s*\n+', '\n', text).strip()
    m = re.search(r'(Date\s*:.*?)Related Press Releases', text, re.S)
    if not m:
        return None
    return m.group(1).strip()

if __name__ == "__main__":
    import os
    os.makedirs("data/rbi_docs/press_releases", exist_ok=True)
    for prid in PRIDS:
        body = fetch_press_release(prid)
        if not body or len(body) < 100:
            print(prid, "SKIPPED - extraction failed")
            continue
        with open(f"data/rbi_docs/press_releases/pr_{prid}.txt", "w", encoding="utf-8") as f:
            f.write(body)
        first_line = body.split("\n")[1][:70] if len(body.split("\n")) > 1 else body[:70]
        print(prid, "->", first_line)