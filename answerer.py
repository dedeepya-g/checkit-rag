import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL_LARGE = "llama-3.3-70b-versatile"
GROQ_MODEL_FAST  = "llama-3.1-8b-instant"

# Keywords that signal a complex query needing the large model
_COMPLEX_PATTERNS = [
    "compare", "comparison", "vs", "versus", "difference",
    "trend", "over time", "history", "growth", "year over year", "yoy",
    "across", "all companies", "multiple", "breakdown", "analysis",
    "why", "explain", "what drove", "what caused",
]


def _pick_model(query: str) -> str:
    q = query.lower()
    if any(p in q for p in _COMPLEX_PATTERNS):
        return GROQ_MODEL_LARGE
    return GROQ_MODEL_FAST

SYSTEM_PROMPT = """You are a senior buy-side financial analyst at Checkit Analytics. \
You think and communicate like an experienced equity research professional — not a data retrieval system.

You have access to SEC filings and earnings press releases for 20 major companies spanning 2020–2026:
- 8-K: earnings press releases with financial results, management commentary, and guidance
- 10-K: annual reports with full-year financials, risk factors, and business overview
- 10-Q: quarterly reports with financial statements and MD&A

## How to respond

**Lead with the insight, not the data.** Open with what actually matters — the key takeaway a \
portfolio manager would want to hear first.

**Interpret, don't just report.** After stating a number, explain what's driving it: mix shift, \
pricing power, cost leverage, macro tailwinds, one-time items, etc. Use language like:
- "Revenue acceleration was primarily driven by..."
- "Margin compression reflects..."
- "Management flagged X as a key risk heading into..."
- "The beat/miss was concentrated in..."

**Use precise financial language.** YoY, QoQ, CAGR, operating leverage, EBIT/EBITDA margins, \
free cash flow conversion, guidance revision, beat-and-raise, multiple compression.

**Format for analysts:**
- Short executive summary (1–2 sentences) at the top
- Structured bullets or a table for comparisons and trends
- Bold the key numbers
- Flag inflection points, beats/misses, and guidance changes explicitly

**Source discipline:**
- Cite inline as (TICKER, Q# YYYY) — e.g. (NVDA, Q3 2024)
- Never paste raw filenames or metadata strings
- If context is incomplete, state exactly what's missing and what you *can* conclude from what's available — never fabricate

**Company-specific queries — STRICT RULE:**
If the user asks about a specific company (e.g. AAPL) and the exact period is not in context, \
you MUST use the nearest available period for THAT SAME COMPANY from the context provided. \
NEVER use data from other companies as a proxy or comparison substitute. \
Always explicitly state: "Exact [requested period] data is unavailable; the nearest available data for [TICKER] is from [actual period]:" \
then answer using that data.

**For follow-ups**, use conversation history to resolve references like "it", "they", "that quarter" \
without asking for clarification."""


@st.cache_resource(ttl=3600)
def _get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env")
    return Groq(api_key=api_key)


def _confidence_level(chunks):
    if not chunks:
        return "low"
    top_score = chunks[0].get("score", 0)
    if top_score > 0.6:
        return "high"
    elif top_score > 0.4:
        return "medium"
    return "low"


CHUNK_CONTEXT_LIMIT = 700  # chars per chunk sent to the LLM


def _build_context(chunks):
    parts = []
    for c in chunks:
        header = f"[{c['ticker']} | {c['form_type']} | {c['quarter']} | {c.get('date', '')}]"
        text = c['text'][:CHUNK_CONTEXT_LIMIT]
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)


def _extract_citations(chunks):
    seen = set()
    citations = []
    for c in chunks:
        key = (c["ticker"], c["quarter"], c["form_type"])
        if key not in seen:
            seen.add(key)
            citations.append({
                "ticker": c["ticker"],
                "quarter": c["quarter"],
                "form_type": c["form_type"],
                "date": c.get("date", ""),
            })
    return citations


def answer(query, chunks, conversation_history=None):
    """
    Generate an answer from retrieved chunks using Groq.

    Args:
        query: user's current question
        chunks: list of dicts from searcher.search()
        conversation_history: list of {"role": "user"/"assistant", "content": str}

    Returns:
        dict with keys: answer, citations, confidence, chunks_used
    """
    client = _get_client()
    confidence = _confidence_level(chunks)
    context = _build_context(chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject prior conversation turns — last 6 messages (3 exchanges) to cap context size
    if conversation_history:
        messages.extend(conversation_history[-6:])

    # Current turn: context + question
    user_message = f"Context from SEC filings:\n{context}\n\nQuestion: {query}"
    messages.append({"role": "user", "content": user_message})

    model = _pick_model(query)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )
    answer_text = response.choices[0].message.content.strip()

    return {
        "answer": answer_text,
        "citations": _extract_citations(chunks),
        "confidence": confidence,
        "chunks_used": len(chunks),
    }


if __name__ == "__main__":
    import searcher
    chunks = searcher.search("Apple revenue Q3 2024")
    result = answer("What was Apple's revenue in Q3 2024?", chunks)
    print(result["answer"])
    print(f"\nConfidence: {result['confidence']} | Chunks used: {result['chunks_used']}")
