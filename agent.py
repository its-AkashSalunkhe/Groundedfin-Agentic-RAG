import os, operator
from typing import TypedDict, List, Tuple, Annotated
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
import requests
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
import re


load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
gnews_key = os.getenv("GNEWS_API_KEY")

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection("fraud_knowledge", embedding_function=ef)
long_term_mem = chroma_client.get_or_create_collection("user_memory", embedding_function=ef)


class AgentState(TypedDict):
    question: str
    standalone_question: str
    query_for_kb: str
    label: str
    kb_results: List[Tuple[str, dict]]
    kb_relevant: bool
    kb_attempts: int
    news_results: List[Tuple[str, str, str]]
    long_term_context: str
    chat_history: Annotated[List[Tuple[str, str]], operator.add]
    answer: str
    user_id: str
    blocked: bool
    grounded: bool
    ground_attempts: int


# ---------- LLM calls ----------

def call_llm(prompt, temperature=0.2):
    resp = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return resp.choices[0].message.content


def contextualize_question(question, chat_history):
    if not chat_history:
        return question
    history_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in chat_history[-3:])
    prompt = f"""Given this recent conversation and a new question, rewrite the new question to be fully self-contained if it depends on prior context. If it's already self-contained, return it unchanged.
Output ONLY the rewritten question text. Do not include any label, prefix, or explanation.

Recent conversation:
{history_text}

New question: {question}
Self-contained question:"""
    result = call_llm(prompt, temperature=0).strip()
    if result.lower().startswith("self-contained question:"):
        result = result.split(":", 1)[1].strip()
    return result


def classify_query(question):
    prompt = f"""Classify this question into exactly one word: STATIC, LIVE, BOTH, DIRECT, or OUT_OF_SCOPE.
STATIC - asking what a fraud or scam type is, how it works, or general precautions — even if the specific term might not be in the knowledge base (the system will search and can fall back to live news)
LIVE - asking about recent news, current events, what's happening now
BOTH - needs both general knowledge and recent news
DIRECT - greeting, thanks, or something needing no lookup at all
OUT_OF_SCOPE - anything clearly unrelated to fraud, scams, or financial safety (recipes, trivia, coding help, unrelated small talk) — not a fraud-related question just because the term is newer or uncommon

Question: {question}
Answer with one word only."""
    return call_llm(prompt, temperature=0).strip().upper()


def extract_search_keywords(question):
    prompt = f"""Extract the core topic from this question as 2-3 keywords for a news search, space-separated, no commas, no quotes.
Do not include words like "recent", "latest", "India", "news", or question words.

Question: {question}
Keywords:"""
    return call_llm(prompt, temperature=0).strip().strip('"').replace(",", "")


def rewrite_kb_query(question):
    prompt = f"""This search query found no good matches in a fraud-awareness knowledge base. Rewrite it using different, broader, or more general terms that might match how the source material describes this topic.

Original query: {question}
Rewritten query:"""
    return call_llm(prompt, temperature=0.3).strip().strip('"')


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


def fetch_kb_with_distance(query, n=5):
    results = collection.query(query_texts=[query], n_results=n)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    best_distance = min(distances) if distances else 1.0
    return list(zip(docs, metas)), best_distance


def fetch_long_term_memory(question, user_id, n=2):
    if long_term_mem.count() == 0:
        return ""
    results = long_term_mem.query(query_texts=[question], n_results=n, where={"user_id": user_id})
    docs = results["documents"][0]
    distances = results["distances"][0]
    print("[debug] memory distances:", distances)
    good = [d for d, dist in zip(docs, distances) if dist < 0.5]
    return "\n".join(good)


def save_long_term_memory(question, answer, user_id):
    summary = f"User asked: {question} | Answer summary: {answer[:200]}"
    existing = long_term_mem.count()
    long_term_mem.add(
        ids=[f"{user_id}_{existing}"],
        documents=[summary],
        metadatas=[{"user_id": user_id}]
    )


INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all |the )?(previous|prior|above)",
    r"you are now",
    r"reveal (your |the )?(system prompt|instructions)",
    r"what (are|is) your (system prompt|instructions)",
    r"pretend (you|to) (are|be)",
    r"jailbreak",
    r"developer mode",
    r"do anything now",
    r"forget (your |all )?(previous )?(instructions|rules)",
]
_INJECTION_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def check_injection(text):
    for pattern in _INJECTION_COMPILED:
        if pattern.search(text):
            return True
    return False


def redact_pii(text):
    text = re.sub(r'\b(?:\d[ -]?){13,19}\b', '[CARD NUMBER REDACTED] ', text)
    text = re.sub(r'\b\d{12}\b', '[ID NUMBER REDACTED]', text)
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL REDACTED]', text)
    text = re.sub(r'(\+91[\-\s]?)?\b[6-9]\d{9}\b', '[PHONE REDACTED]', text)
    text = re.sub(r'\b(OTP|PIN|CVV)\s*(is|:)?\s*\d{3,6}\b', r'\1 [REDACTED]', text, flags=re.IGNORECASE)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


# ---------- graph nodes ----------

def input_guardrail_node(state):
    if check_injection(state["question"]):
        return {"blocked": True, "answer": "I can't follow instructions embedded in a question like that. Feel free to ask me anything about fraud and scam awareness directly."}
    return {"blocked": False}


def contextualize_node(state):
    sq = contextualize_question(state["question"], state.get("chat_history", []))
    return {"standalone_question": sq}


def recall_node(state):
    ctx = fetch_long_term_memory(state["standalone_question"], state["user_id"])
    return {"long_term_context": ctx}


def classify_node(state):
    return {"label": classify_query(state["standalone_question"])}


def out_of_scope_node(state):
    answer = "I'm built specifically for fraud and scam awareness in India. I can't help with that, but ask me anything about phishing, vishing, card fraud, or similar topics."
    return {"answer": answer, "chat_history": [(state["question"], answer)]}


def kb_node(state):
    attempt = state.get("kb_attempts", 0)
    query = state.get("query_for_kb") or state["standalone_question"]
    results, best_distance = fetch_kb_with_distance(query)
    relevant = best_distance < 0.6
    print(f"[debug] kb attempt={attempt} best_distance={best_distance:.3f} relevant={relevant}")
    if relevant:
        for doc, meta in results:
            print(f"[debug]   retrieved: {doc[:60]!r} (source: {meta.get('source')})")
    return {
        "kb_results": results if relevant else [],
        "kb_relevant": relevant,
        "kb_attempts": attempt + 1,
    }

def kb_rewrite_node(state):
    new_query = rewrite_kb_query(state["standalone_question"])
    print(f"[debug] kb rewrite -> '{new_query}'")
    return {"query_for_kb": new_query}


def news_node(state):
    return {"news_results": fetch_live_news(state["standalone_question"])}


def direct_node(state):
    prompt = f"""You are GroundedFin, a helpful assistant focused on fraud and scam awareness in India. Respond naturally and briefly.

Message: {state['question']}
Answer:"""
    answer = call_llm(prompt)
    return {"answer": answer, "chat_history": [(state["question"], answer)]}


def build_context(state, include_memory=True):
    kb_results = state.get("kb_results", [])
    news_results = state.get("news_results", [])
    long_term_context = state.get("long_term_context", "")
    context = ""
    if include_memory and long_term_context:
        context += f"What you remember about this user from past conversations:\n{long_term_context}\n\n"
    if kb_results:
        context += "From RBI fraud awareness material:\n"
        for doc, meta in kb_results:
            context += f"- [{meta['source']}] {doc[:500]}\n"
    if news_results:
        context += "\nRecent news:\n"
        for title, date, url in news_results:
            context += f"- {title} ({date})\n"
    return context


def generate_node(state):
    kb_results = state.get("kb_results", [])
    news_results = state.get("news_results", [])
    has_real_content = bool(kb_results or news_results)
    attempt = state.get("ground_attempts", 0)

    if not has_real_content:
        return {"answer": ("I couldn't find reliable information on this in the fraud knowledge base or recent news "
                            "results. For the latest guidance, check RBI's official channels or report at "
                            "cybercrime.gov.in / helpline 1930.")}

    # on the stricter retry, drop memory from context entirely -- the model can't misuse
    # what it can no longer see, and this matches exactly what groundedness checks against
    context = build_context(state, include_memory=(attempt == 0))
    strictness = "" if attempt == 0 else (
        "\n\nIMPORTANT: Your previous answer included claims not present in the context. "
        "This time, state ONLY facts that are explicitly present in the context below. "
        "Do not add examples, numbers, or details you are inferring or recalling from general knowledge. "
        "If recent news doesn't cover the question, say so plainly."
    )
    prompt = f"""Answer the question using only the context below. Cite the source in plain parentheses like (Source: Times of India, 26 Sep 2026), not special brackets. Use the "what you remember" section only to personalize tone, never as a factual source. Never describe something as recent unless its source date is within the last 30 days. If the context doesn't fully cover the question, say so honestly.{strictness}

Context:
{context}

Question: {state['standalone_question']}
Answer:"""
    answer = call_llm(prompt)
    return {"answer": answer}


def groundedness_check_node(state):
    kb_results = state.get("kb_results", [])
    news_results = state.get("news_results", [])
    has_real_content = bool(kb_results or news_results)
    attempt = state.get("ground_attempts", 0)

    if not has_real_content:
        return {"grounded": True, "ground_attempts": attempt}

    context = build_context(state, include_memory=False)
    prompt = f"""You are checking an AI-generated answer for accuracy.

Rules:
- If the ENTIRE answer simply states that no relevant information is available, or expresses inability to answer due to missing context, with no additional specific facts, dates, or figures beyond that -- this is automatically GROUNDED. Answer YES.
- If the answer includes ANY specific fact, number, date, or named detail, that detail must be explicitly present in the context below. If even one such detail is not explicitly present in the context, answer NO.

Context:
{context}

Answer to check:
{state['answer']}

Is this answer grounded according to the rules above? YES or NO:"""
    verdict = call_llm(prompt, temperature=0).strip().upper()
    grounded = verdict.startswith("YES")
    print(f"[debug] groundedness attempt={attempt} verdict={verdict} grounded={grounded}")
    return {"grounded": grounded, "ground_attempts": attempt + 1}


def finalize_node(state):
    answer = state["answer"]
    kb_results = state.get("kb_results", [])
    news_results = state.get("news_results", [])
    has_real_content = bool(kb_results or news_results)

    if has_real_content and not state["grounded"]:
        answer += " [Note: some details in this answer could not be fully verified against the source material.]"

    answer = redact_pii(answer)
    if has_real_content:
        save_long_term_memory(redact_pii(state["question"]), answer, state["user_id"])
    return {"answer": answer, "chat_history": [(state["question"], answer)]}


def route_after_ground(state):
    if state["grounded"] or state["ground_attempts"] >= 2:
        return "finalize"
    return "regenerate"


# ---------- routing ----------

def route_after_classify(state):
    label = state["label"]
    if label == "DIRECT":
        return "direct"
    elif label == "OUT_OF_SCOPE":
        return "out_of_scope"
    elif label == "STATIC":
        return "kb"
    elif label == "LIVE":
        return "news"
    else:
        return "kb_then_news"


def route_after_kb(state):
    if state["kb_relevant"]:
        return "news" if state["label"] == "BOTH" else "generate"
    if state["kb_attempts"] >= 2:
        return "news"  # KB found nothing after retries -- check live news before giving up
    return "rewrite"


# ---------- build graph ----------

graph = StateGraph(AgentState)
graph.add_node("contextualize", contextualize_node)
graph.add_node("input_guardrail", input_guardrail_node)
graph.add_node("recall", recall_node)
graph.add_node("classify", classify_node)
graph.add_node("kb", kb_node)
graph.add_node("kb_rewrite", kb_rewrite_node)
graph.add_node("news", news_node)
graph.add_node("direct", direct_node)
graph.add_node("out_of_scope", out_of_scope_node)
graph.add_node("generate", generate_node)
graph.add_node("ground_check", groundedness_check_node)
graph.add_node("finalize", finalize_node)

graph.add_edge(START, "input_guardrail")
graph.add_conditional_edges("input_guardrail", lambda s: "blocked" if s["blocked"] else "continue", {
    "blocked": END,
    "continue": "contextualize",
})
graph.add_edge("contextualize", "recall")
graph.add_edge("recall", "classify")
graph.add_conditional_edges("classify", route_after_classify, {
    "direct": "direct",
    "out_of_scope": "out_of_scope",
    "kb": "kb",
    "news": "news",
    "kb_then_news": "kb",
})
graph.add_conditional_edges("kb", route_after_kb, {
    "rewrite": "kb_rewrite",
    "news": "news",
    "generate": "generate",
})
graph.add_edge("kb_rewrite", "kb")
graph.add_edge("news", "generate")
graph.add_edge("direct", END)
graph.add_edge("out_of_scope", END)
graph.add_edge("generate", "ground_check")
graph.add_conditional_edges("ground_check", route_after_ground, {
    "finalize": "finalize",
    "regenerate": "generate",
})
graph.add_edge("finalize", END)

app = graph.compile(checkpointer=InMemorySaver())


def ask(question, user_id="default_user", thread_id="default_thread"):
    cfg = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({
        "question": question,
        "user_id": user_id,
        "grounded": False,
        "ground_attempts": 0,
        "kb_attempts": 0,
        "kb_relevant": False,
        "query_for_kb": "",
        "kb_results": [],
        "news_results": [],
        "long_term_context": "",
        "label": "",
        "standalone_question": "",
    }, cfg)
    print(f"[classified as: {result['label']}] [standalone: {result['standalone_question']}]")
    return result["answer"]


if __name__ == "__main__":
    print("Enter a user id to simulate returning users (e.g. 'akash'), or press enter for default.")
    uid = input("user_id: ").strip() or "default_user"
    while True:
        q = input("\nAsk about fraud/scams (or 'quit'): ")
        if q.lower() == "quit":
            break
        print(ask(q, user_id=uid))