import csv, time, os
from dotenv import load_dotenv
from groq import Groq
from agent import fetch_kb_with_distance, fetch_live_news, build_context


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
JUDGE_MODEL = "openai/gpt-oss-20b"  # kept constant, matches your production groundedness check


def call(model, prompt, temperature=0.2):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return resp.choices[0].message.content


def judge_groundedness(answer, context):
    prompt = f"""You are checking an AI-generated answer for accuracy.

Rules:
- If the ENTIRE answer simply states that no relevant information is available, with no additional specific facts, dates, or figures -- this is automatically GROUNDED. Answer YES.
- If the answer includes ANY specific fact, number, date, or named detail, that detail must be explicitly present in the context below. If even one such detail is not explicitly present, answer NO.

Context:
{context}

Answer to check:
{answer}

Is this answer grounded according to the rules above? YES or NO:"""
    verdict = call(JUDGE_MODEL, prompt, temperature=0).strip().upper()
    return verdict.startswith("YES")


with open("test_set.csv") as f:
    all_rows = {r["id"]: r for r in csv.DictReader(f)}

QUESTION_IDS = ["S1", "S3", "S5", "L1", "L3", "B1", "B3", "T1"]
rows = [all_rows[qid] for qid in QUESTION_IDS]

results = []
for row in rows:
    question = row["question"]
    print(f"\n--- {row['id']}: {question} ---")

    fake_state = {"standalone_question": question, "kb_results": [], "news_results": []}
    if row["category"] in ("STATIC", "BOTH", "TYPO"):
        kb_results, _ = fetch_kb_with_distance(question)
        fake_state["kb_results"] = kb_results
    if row["category"] in ("LIVE", "BOTH"):
        fake_state["news_results"] = fetch_live_news(question)

    context = build_context(fake_state, include_memory=False)

    for model in MODELS:
        prompt = f"""Answer the question using only the context below. Cite sources in plain parentheses.

Context:
{context}

Question: {question}
Answer:"""
        start = time.time()
        answer = call(model, prompt)
        elapsed = time.time() - start
        grounded = judge_groundedness(answer, context)
        results.append({
            "id": row["id"], "model": model, "grounded": grounded,
            "time_sec": round(elapsed, 2), "answer_len": len(answer), "answer": answer
        })
        print(f"  {model}: grounded={grounded} time={elapsed:.2f}s len={len(answer)}")
        time.sleep(2)

print("\n=== Summary by model ===")
by_model = {}
for r in results:
    by_model.setdefault(r["model"], []).append(r)
for model, rs in by_model.items():
    grounded_count = sum(1 for r in rs if r["grounded"])
    avg_time = sum(r["time_sec"] for r in rs) / len(rs)
    avg_len = sum(r["answer_len"] for r in rs) / len(rs)
    print(f"{model}: {grounded_count}/{len(rs)} grounded | avg {avg_time:.2f}s | avg {avg_len:.0f} chars")

with open("model_comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "model", "grounded", "time_sec", "answer_len", "answer"])
    writer.writeheader()
    writer.writerows(results)
print("\nSaved to model_comparison.csv")