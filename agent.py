import os, json
from typing import TypedDict
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
import requests

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
gnews_key = os.getenv("GNEWS_API_KEY")

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection("fraud_knowledge", embedding_function=ef)


def classify_query(question):
    prompt = f"""Classify this question into exactly one word: STATIC, LIVE, BOTH, or DIRECT.
STATIC - asking what a scam is, how it works, general precautions (answerable from a fraud awareness booklet)
LIVE - asking about recent news, current events, what's happening now
BOTH - needs both general knowledge and recent news
DIRECT - greeting, thanks, or something needing no lookup at all

Question: {question}
Answer with one word only."""
    resp = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip().upper()


def search_knowledge_base(question, n=3):
    results = collection.query(query_texts=[question], n_results=n)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return list(zip(docs, metas))


def extract_search_keywords(question):
    prompt = f"""Extract the core topic from this question as 2-3 keywords for a news search, space-separated, no commas, no quotes.
Do not include words like "recent", "latest", "India", "news", or question words.

Question: {question}
Keywords:"""
    resp = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip().strip('"').replace(",", "")


def _gnews_query(q, max_results):
    url = "https://gnews.io/api/v4/search"
    params = {"q": q, "lang": "en", "country": "in", "max": max_results, "sortby": "publishedAt", "apikey": gnews_key}
    r = requests.get(url, params=params, timeout=10)
    return r.json().get("articles", [])



def search_live_news(question, max_results=5):
    keywords = extract_search_keywords(question)
    print(f"[debug] news search keywords: {keywords}")
    articles = _gnews_query(keywords, max_results)

    if not articles:
        # fallback 1: drop the country filter, in case it's over-narrowing
        url = "https://gnews.io/api/v4/search"
        params = {"q": keywords, "lang": "en", "max": max_results, "sortby": "publishedAt", "apikey": gnews_key}
        r = requests.get(url, params=params, timeout=10)
        articles = r.json().get("articles", [])
        print(f"[debug] fallback (no country filter) results: {len(articles)}")

    if not articles:
        # fallback 2: just the first keyword, broadest possible search
        broad_term = keywords.split()[0] if keywords else question.split()[0]
        articles = _gnews_query(broad_term, max_results)
        print(f"[debug] fallback (broad term '{broad_term}') results: {len(articles)}")

    return [(a["title"], a["publishedAt"], a["url"]) for a in articles]


def generate_answer(question, kb_results=None, news_results=None, mode="grounded"):
    context = ""
    if kb_results:
        context += "From RBI fraud awareness material:\n"
        for doc, meta in kb_results:
            context += f"- [{meta['source']}] {doc[:500]}\n"
    if news_results:
        context += "\nRecent news:\n"
        for title, date, url in news_results:
            context += f"- {title} ({date})\n"

    if mode == "direct":
        prompt = f"""You are GroundedFin, a helpful assistant focused on fraud and scam awareness in India. Respond naturally and briefly. If it's a greeting or small talk, just reply normally.

Message: {question}
Answer:"""
    elif context:
        prompt = f"""Answer the question using only the context below. Cite the source (booklet section or news article) for each claim. If the context doesn't fully cover the question, say so honestly rather than guessing.

Context:
{context}

Question: {question}
Answer:"""
    else:
        return ("I couldn't find reliable information on this in the fraud knowledge base or recent news results. "
                "For the latest guidance, check RBI's official channels or report at cybercrime.gov.in / helpline 1930.")

    resp = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content


def ask(question):
    label = classify_query(question)
    print(f"[classified as: {label}]")

    if label == "DIRECT":
        return generate_answer(question, mode="direct")
    elif label == "STATIC":
        kb = search_knowledge_base(question)
        return generate_answer(question, kb_results=kb)
    elif label == "LIVE":
        news = search_live_news(question)
        print(f"[debug] news results: {news}")
        return generate_answer(question, news_results=news)
    else:
        kb = search_knowledge_base(question)
        news = search_live_news(question)
        print(f"[debug] news results: {news}")
        return generate_answer(question, kb_results=kb, news_results=news)


if __name__ == "__main__":
    while True:
        q = input("\nAsk about fraud/scams (or 'quit'): ")
        if q.lower() == "quit":
            break
        print(ask(q))