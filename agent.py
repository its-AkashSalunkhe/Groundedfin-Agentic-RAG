import os
from typing import TypedDict, List, Tuple, Optional
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
import requests
from langgraph.graph import StateGraph, START, END

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
gnews_key = os.getenv("GNEWS_API_KEY")

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection("fraud_knowledge", embedding_function=ef)


class AgentState(TypedDict):
    question: str
    label: str
    kb_results: List[Tuple[str, dict]]
    news_results: List[Tuple[str, str, str]]
    answer: str


# ---------- LLM calls ----------

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


def _gnews_query(q, max_results, country=None):
    url = "https://gnews.io/api/v4/search"
    params = {"q": q, "lang": "en", "max": max_results, "sortby": "publishedAt", "apikey": gnews_key}
    if country:
        params["country"] = country
    r = requests.get(url, params=params, timeout=10)
    return r.json().get("articles", [])


def fetch_live_news(question, max_results=5):
    keywords = extract_search_keywords(question)
    articles = _gnews_query(keywords, max_results, country="in")
    if not articles:
        articles = _gnews_query(keywords, max_results)
    if not articles:
        broad_term = keywords.split()[0] if keywords else question.split()[0]
        articles = _gnews_query(broad_term, max_results)
    return [(a["title"], a["publishedAt"], a["url"]) for a in articles]


def fetch_kb(question, n=3):
    results = collection.query(query_texts=[question], n_results=n)
    return list(zip(results["documents"][0], results["metadatas"][0]))


def call_llm(prompt, temperature=0.2):
    resp = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return resp.choices[0].message.content


# ---------- graph nodes ----------

def classify_node(state):
    return {"label": classify_query(state["question"])}


def kb_node(state):
    return {"kb_results": fetch_kb(state["question"])}


def news_node(state):
    return {"news_results": fetch_live_news(state["question"])}


def direct_node(state):
    prompt = f"""You are GroundedFin, a helpful assistant focused on fraud and scam awareness in India. Respond naturally and briefly. If it's a greeting or small talk, just reply normally.

Message: {state['question']}
Answer:"""
    return {"answer": call_llm(prompt)}


def generate_node(state):
    kb_results = state.get("kb_results", [])
    news_results = state.get("news_results", [])
    context = ""
    if kb_results:
        context += "From RBI fraud awareness material:\n"
        for doc, meta in kb_results:
            context += f"- [{meta['source']}] {doc[:500]}\n"
    if news_results:
        context += "\nRecent news:\n"
        for title, date, url in news_results:
            context += f"- {title} ({date})\n"

    if not context:
        return {"answer": ("I couldn't find reliable information on this in the fraud knowledge base or recent "
                            "news results. For the latest guidance, check RBI's official channels or report at "
                            "cybercrime.gov.in / helpline 1930.")}

    prompt = f"""Answer the question using only the context below. Cite the source in plain parentheses like (Source: Times of India, 26 Sep 2026), not special brackets. If the context doesn't fully cover the question, say so honestly rather than guessing.

Context:
{context}

Question: {state['question']}
Answer:"""
    return {"answer": call_llm(prompt)}


# ---------- routing ----------

def route_after_classify(state):
    label = state["label"]
    if label == "DIRECT":
        return "direct"
    elif label == "STATIC":
        return "kb"
    elif label == "LIVE":
        return "news"
    else:
        return "kb_then_news"


def route_after_kb(state):
    return "news" if state["label"] == "BOTH" else "generate"


# ---------- build graph ----------

graph = StateGraph(AgentState)
graph.add_node("classify", classify_node)
graph.add_node("kb", kb_node)
graph.add_node("news", news_node)
graph.add_node("direct", direct_node)
graph.add_node("generate", generate_node)

graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route_after_classify, {
    "direct": "direct",
    "kb": "kb",
    "news": "news",
    "kb_then_news": "kb",
})
graph.add_conditional_edges("kb", route_after_kb, {"news": "news", "generate": "generate"})
graph.add_edge("news", "generate")
graph.add_edge("direct", END)
graph.add_edge("generate", END)

app = graph.compile()


def ask(question):
    result = app.invoke({
        "question": question, "label": "", "kb_results": [], "news_results": [], "answer": ""
    })
    print(f"[classified as: {result['label']}]")
    return result["answer"]


if __name__ == "__main__":
    while True:
        q = input("\nAsk about fraud/scams (or 'quit'): ")
        if q.lower() == "quit":
            break
        print(ask(q))