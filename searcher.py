import os
import re
import hashlib
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from collections import Counter

load_dotenv()

INDEX_NAME = "checkit-rag"
EMBED_MODEL = "multilingual-e5-large"
LOW_CONFIDENCE_THRESHOLD = 0.35
RERANK_TOP_K = 6       # chunks sent to LLM after reranking
SEMANTIC_WEIGHT = 0.6  # weight for semantic score in fusion
BM25_WEIGHT = 0.4      # weight for BM25 score in fusion

# Minimum chunks from the mentioned ticker before triggering a fallback search
FALLBACK_THRESHOLD = 2

KNOWN_TICKERS = {
    "AAPL", "MSFT", "META", "NVDA", "TSLA",
    "JPM", "GS", "SBUX", "WMT", "MCD",
    "NKE", "DIS", "CRM", "GOOGL", "AMZN",
    "NFLX", "ORCL", "MS", "BAC", "V",
}

# Quarter/year patterns to strip when building a fallback query
_PERIOD_RE = re.compile(
    r"\b(Q[1-4]\s*\d{4}|\d{4}\s*Q[1-4]|Q[1-4]|FY\s*\d{2,4}|fiscal\s+\d{4}|\d{4})\b",
    re.IGNORECASE,
)


@st.cache_resource(ttl=3600)
def _get_pinecone():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set in .env")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)
    return pc, index


def _token_to_idx(token: str) -> int:
    """Convert a token string to a stable non-negative integer for Pinecone sparse vectors."""
    return int(hashlib.md5(token.encode()).hexdigest()[:8], 16)


def _extract_ticker(query: str):
    """Return the first known ticker mentioned in the query, or None."""
    # Replace non-alpha chars with spaces so "AAPL's" → "AAPL s", not "AAPLS"
    tokens = re.sub(r"[^A-Z]", " ", query.upper()).split()
    for token in tokens:
        if token in KNOWN_TICKERS:
            return token
    # Also check full company name aliases
    aliases = {
        "APPLE": "AAPL", "MICROSOFT": "MSFT", "AMAZON": "AMZN",
        "GOOGLE": "GOOGL", "ALPHABET": "GOOGL", "NVIDIA": "NVDA",
        "TESLA": "TSLA", "NETFLIX": "NFLX", "STARBUCKS": "SBUX",
        "WALMART": "WMT", "NIKE": "NKE", "DISNEY": "DIS",
        "SALESFORCE": "CRM", "ORACLE": "ORCL", "META": "META",
        "GOLDMAN": "GS", "JPMORGAN": "JPM", "MORGAN STANLEY": "MS",
        "BANK OF AMERICA": "BAC", "VISA": "V", "MCDONALDS": "MCD",
    }
    q_upper = query.upper()
    for name, ticker in aliases.items():
        if name in q_upper:
            return ticker
    return None


@st.cache_data(ttl=300)
def search(query, ticker_filter=None, form_type_filter=None, top_k=20, min_score=0.30,
           date_start=None, date_end=None):
    """
    Search Pinecone for relevant chunks, rerank, and return top RERANK_TOP_K results.

    Args:
        query:            search string
        ticker_filter:    tuple of ticker strings e.g. ("AAPL", "MSFT"), or None for all
        form_type_filter: "8-K", "10-K", "10-Q", or None for all
        top_k:            candidates to fetch from Pinecone before reranking
        min_score:        minimum semantic score to keep a candidate
        date_start:       ISO date string lower bound e.g. "2022-01-01"
        date_end:         ISO date string upper bound e.g. "2024-12-31"

    Returns:
        list of result dicts (up to RERANK_TOP_K) sorted by fusion score
    """
    pc, index = _get_pinecone()

    # ── Dense embedding ───────────────────────────────────────────────────────
    embed_result = pc.inference.embed(
        model=EMBED_MODEL,
        inputs=[query],
        parameters={"input_type": "query", "truncate": "END"},
    )
    embedding = embed_result[0]["values"]

    # ── BM25 sparse query vector (term-frequency, integer indices) ───────────
    # Pinecone requires integer indices; use a stable MD5 hash per token so
    # query-time and index-time vocabulary are consistent after re-indexing.
    query_tokens = query.lower().split()
    tf = Counter(query_tokens)
    sparse_vector = {
        "indices": [_token_to_idx(t) for t in tf.keys()],
        "values":  [float(v) for v in tf.values()],
    }

    # ── Metadata filter ───────────────────────────────────────────────────────
    conditions = []
    if ticker_filter:
        conditions.append({"ticker": {"$in": list(ticker_filter)}})
    if form_type_filter and form_type_filter != "All":
        conditions.append({"form_type": {"$eq": form_type_filter}})
    if date_start:
        conditions.append({"date": {"$gte": date_start}})
    if date_end:
        conditions.append({"date": {"$lte": date_end}})

    pinecone_filter = (
        conditions[0] if len(conditions) == 1
        else {"$and": conditions} if conditions
        else {}
    )

    kwargs = {
        "vector": embedding,
        "sparse_vector": sparse_vector,
        "top_k": top_k,
        "include_metadata": True,
    }
    if pinecone_filter:
        kwargs["filter"] = pinecone_filter

    # ── Query Pinecone (hybrid → dense fallback) ──────────────────────────────
    try:
        response = index.query(**kwargs)
    except Exception as e:
        if "sparse" in str(e).lower() or "dotproduct" in str(e).lower():
            kwargs.pop("sparse_vector")
            response = index.query(**kwargs)
        else:
            raise

    # ── Collect candidates above min_score ───────────────────────────────────
    candidates = []
    for match in response.matches:
        if match.score < min_score:
            continue
        meta = match.metadata or {}
        candidates.append({
            "score":          round(match.score, 4),
            "low_confidence": match.score < LOW_CONFIDENCE_THRESHOLD,
            "ticker":         meta.get("ticker", ""),
            "form_type":      meta.get("form_type", ""),
            "date":           meta.get("date", ""),
            "quarter":        meta.get("quarter", ""),
            "speaker":        meta.get("speaker", "N/A"),
            "text":           meta.get("text", ""),
        })

    return _rerank(query, candidates)


@st.cache_data(ttl=300)
def search_with_fallback(query, ticker_filter=None, form_type_filter=None,
                         date_start=None, date_end=None):
    """
    Search with automatic same-company fallback.

    If the query mentions a specific ticker but the initial results contain
    fewer than FALLBACK_THRESHOLD chunks for that company, a second search is
    run for that company only — stripping the period constraint so the nearest
    available data is returned instead of drifting to other companies.
    """
    results = search(
        query,
        ticker_filter=ticker_filter,
        form_type_filter=form_type_filter,
        date_start=date_start,
        date_end=date_end,
    )

    mentioned_ticker = _extract_ticker(query)
    if not mentioned_ticker:
        return results

    ticker_hits = [r for r in results if r["ticker"] == mentioned_ticker]
    if len(ticker_hits) >= FALLBACK_THRESHOLD:
        return results  # enough company-specific data — no fallback needed

    # ── Fallback: same company, no period constraint, broadened query ─────────
    # Strip quarter/year tokens so we don't search for a period we don't have
    broad_query = _PERIOD_RE.sub("", query).strip()
    if not broad_query or broad_query.upper() == mentioned_ticker:
        broad_query = f"{mentioned_ticker} revenue earnings financial results"

    fallback = search(
        broad_query,
        ticker_filter=(mentioned_ticker,),  # lock to this company only
        form_type_filter=form_type_filter,
        date_start=None,  # remove date constraints — take whatever is nearest
        date_end=None,
    )

    return fallback if fallback else results


def _rerank(query, candidates):
    """
    Fuse semantic scores with BM25 keyword scores.
    Returns top RERANK_TOP_K chunks sorted by fusion score.
    Does NOT mutate the input dicts.
    """
    if not candidates:
        return candidates

    tokenised = [c["text"].lower().split() for c in candidates]
    bm25 = BM25Okapi(tokenised)
    bm25_scores = bm25.get_scores(query.lower().split())

    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    norm_bm25 = [s / max_bm25 for s in bm25_scores]

    scored = []
    for i, c in enumerate(candidates):
        fusion = SEMANTIC_WEIGHT * c["score"] + BM25_WEIGHT * norm_bm25[i]
        scored.append((round(fusion, 4), c))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for fusion_score, c in scored[:RERANK_TOP_K]:
        results.append({**c, "score": fusion_score})

    return results


if __name__ == "__main__":
    results = search_with_fallback("Apple revenue Q3 2024")
    for r in results:
        print(f"[{r['score']:.3f}] {r['ticker']} {r['quarter']} {r['speaker']}")
        print(f"  {r['text'][:120]}...")
        print()
