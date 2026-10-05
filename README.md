# GroundedFin

A trustworthy, agentic RAG system for fraud and scam awareness in India — combining RBI's official fraud-awareness material with live news, built with LangGraph, ChromaDB, and Groq.

Ask it "what is vishing?" and it answers from RBI's official guidance. Ask it "any recent UPI fraud cases?" and it checks live news. Ask it something out of scope, and it says so cleanly instead of guessing. And before any answer reaches you, a second pass checks whether it's actually backed by the source material it was given — catching the kind of hallucination that plagues naive RAG systems.

## Why this project

Most portfolio RAG projects retrieve-then-generate and stop there. GroundedFin treats that as the easy 80%, and spends the remaining effort on the part that's usually missing: guardrails and an eval harness that actually measures whether the system is trustworthy, not just functional.

## Architecture


Built as a LangGraph `StateGraph` with typed state and two retry loops (KB retrieval, answer groundedness), not a flat prompt chain.

## Data sources

- **RBI's "BE(A)WARE" booklet** — 36-page official fraud-awareness document covering 20 fraud types (phishing, vishing, SIM swap, money mules, loan app scams, and more), chunked by section for precise retrieval.
- **9 RBI press releases** (2007–2020) on specific fraud and phishing warnings, scraped directly from RBI's site.
- **GNews API** — live news, filtered and re-queried with progressively broader terms when a search comes back empty.

## Guardrails

| Guardrail | How it works |
|---|---|
| Prompt injection | Rule-based pattern matching on the raw input, before any LLM call touches it |
| PII redaction | Regex-based redaction of card numbers, Aadhaar-style IDs, emails, phone numbers, OTPs — applied to both the answer and anything written to memory |
| Out-of-scope refusal | A dedicated classification label with a fixed response, so refusal is deliberate, not an accidental side effect of weak retrieval |
| Groundedness check | A second LLM call verifies every claim in the answer traces back to retrieved context; if not, the agent regenerates once with a stricter prompt, and appends an honest caveat if it still can't fully verify |

Deterministic checks run first and are cheap; LLM-based checks are used only where rules genuinely can't do the job.

## Memory

- **Short-term**: per-conversation chat history, carried through LangGraph's checkpointer, used to resolve follow-ups like "any updates on that?"
- **Long-term**: a separate ChromaDB collection, keyed by user ID, that persists across sessions. Both the similarity cutoff for recall (0.5) and for KB retrieval relevance (0.6) were chosen empirically — by measuring real distance scores against known-relevant and known-irrelevant questions — rather than guessed.

## Eval harness

A 46-question single-turn test set across 8 categories (core fraud terms, live news, combined queries, greetings, out-of-scope, prompt injection, PII, and typo/rephrasing robustness), plus 4 multi-turn conversation sequences testing follow-up resolution.

**Latest full run: 51/54 (94%)**

| Category | Result |
|---|---|
| STATIC | 10/10 |
| LIVE | 6/7* |
| BOTH | 4/5 |
| DIRECT | 5/5 |
| OUT_OF_SCOPE | 5/5 |
| Prompt injection | 6/6 |
| PII redaction | 4/4 |
| Typo robustness | 3/4 |

\* one failure was a Groq rate-limit crash mid-run, not an agent error.

Both real failures are understood, not mysteries:
- One typo ("vising" for "vishing") was severe enough to get misclassified as out-of-scope *before* the query-rewrite retry loop ever got a chance to help — a reminder that classification happens upstream of retrieval, so a bad enough typo can skip the fix entirely.
- One compound question ("what is X and is it happening recently?") classified differently across repeated runs of the identical question. This turned out to be a genuine, known limitation of LLM inference, not a bug: `temperature=0` reduces randomness but doesn't guarantee perfect determinism, especially for borderline questions sitting right on a category boundary.

## Notable bugs found and fixed along the way

This project surfaced a lot of real, instructive bugs — documenting them here because catching and understanding them was as much the point as the final result:

- **Hallucinated helpline number**: an early version invented a fake fraud-helpline number when live news came back empty, instead of saying it didn't know. Fixed first by tightening the live-search fallback logic, then more fundamentally by adding the groundedness check — which has since caught and corrected similar issues live, including during a real eval run.
- **Silent edges**: adding graph edges *after* `graph.compile()` doesn't error — LangGraph just quietly ignores them, printing an easy-to-miss warning. An entire guardrail step (the groundedness check) silently never ran until this was caught.
- **State that silently carried over between turns**: a retry counter wasn't reset between separate user questions, since LangGraph's checkpointer preserves any field you don't explicitly overwrite. After enough turns, the retry logic would "max out" before it even started.
- **An unclosed triple-quoted string** swallowed an entire function definition into a string literal, several lines after the actual syntax error — a reminder that Python's error location isn't always where the real mistake is.
- **RBI's document host sat behind a CAPTCHA**, while RBI's plain press-release pages didn't — pivoted the scraping approach entirely rather than trying to force the blocked route.
- **Free-tier infrastructure limits are real constraints**: GNews's free tier has a 12-hour delay on "live" news and a 30-day historical cutoff; Groq's free tier caps at 200K tokens/day, which the groundedness check (which roughly doubles per-answer token cost, since it re-sends full context to verify) can exhaust in a single thorough eval run.

## Known limitations

- Classification is not perfectly deterministic, particularly for compound questions near a category boundary.
- A severe enough typo can be misclassified as out-of-scope before the retry/rewrite logic has a chance to help.
- Near-duplicate long-term memories (e.g., the same question asked with slightly different phrasing) aren't currently deduplicated.
- The free-tier daily token budget limits how often a full eval run can be repeated in one day.
- No UI yet (in progress — Gradio); not yet deployed.

## Tech stack

- **Orchestration**: LangGraph (`StateGraph`, checkpointed memory, conditional routing)
- **LLM**: Groq (`openai/gpt-oss-20b`)
- **Vector store**: ChromaDB (local, persistent)
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Live data**: GNews API
- **Language**: Python

## Setup

```bash
git clone https://github.com/your-username/groundedfin-agentic-rag.git
cd groundedfin-agentic-rag
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Create a `.env` file:



```bash
python ingest.py        # builds the Chroma knowledge base from data/rbi_docs/
python agent.py         # interactive CLI
python run_evals.py     # runs the eval suite
```
## Project structure

## What's next

- Gradio UI
- Deployment
- Demo video
