import os
import requests
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

TICKERS = [
    "AAPL", "MSFT", "META", "NVDA", "TSLA", "JPM", "GS",
    "SBUX", "WMT", "MCD", "NKE", "DIS", "CRM", "GOOGL",
    "AMZN", "NFLX", "ORCL", "MS", "BAC", "V",
]

FALLBACK_QUESTIONS = [
    "What guidance did NVDA give in their latest earnings call?",
    "Compare AAPL and MSFT operating margins from 10-K filings",
    "What risk factors did TSLA flag in their most recent 10-K?",
    "META revenue growth trend from 2022 to 2024 earnings reports",
]

HEADERS = {"User-Agent": "CheckitRAG dg4489@nyu.edu"}


def _fetch_headlines(max_per_ticker=2):
    """Fetch recent Yahoo Finance RSS headlines for our tickers."""
    headlines = []
    # Only sample a subset to keep it fast (pick 8 tickers)
    sample = TICKERS[:8]
    for ticker in sample:
        try:
            url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
            resp = requests.get(url, headers=HEADERS, timeout=5)
            if resp.status_code != 200:
                continue
            # Simple XML parse — extract <title> tags
            import re
            titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", resp.text)
            # Skip the first (feed title) and take next max_per_ticker
            _EARNINGS_KEYWORDS = {
                "earnings", "revenue", "profit", "eps", "beats", "misses",
                "guidance", "outlook", "forecast", "quarter", "quarterly",
                "annual", "10-k", "10-q", "8-k", "sec", "filing",
                "results", "growth", "margin", "sales", "income",
            }
            for t in titles[1:1 + max_per_ticker]:
                if any(kw in t.lower() for kw in _EARNINGS_KEYWORDS):
                    headlines.append(f"{ticker}: {t}")
        except Exception:
            continue
    return headlines


def _generate_questions(headlines):
    """Use Groq to generate 4 trending questions from headlines."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return FALLBACK_QUESTIONS

    client = Groq(api_key=api_key)
    if not headlines:
        return FALLBACK_QUESTIONS
    headline_text = "\n".join(headlines)

    prompt = f"""You are a financial analyst assistant for Checkit Analytics.
We have a RAG system containing SEC filings (10-K annual reports, 10-Q quarterly reports) \
and 8-K earnings press releases for these companies: {', '.join(TICKERS)}.

Based on the recent earnings news headlines below, generate exactly 4 example questions \
that an equity analyst would ask our system. Each question MUST:
- Be directly answerable from an earnings call, 10-K, 10-Q, or 8-K filing
- Focus on one of: earnings guidance, revenue/margin beats or misses, \
management commentary, YoY or QoQ trends, risk factors, or segment performance
- Name a specific company from our list
- Be 8–14 words — analytical and specific, not generic
- NOT ask about stock price, valuation multiples, or analyst ratings

Vary the four questions across these types (one each):
1. Guidance or outlook question (e.g. "What full-year revenue guidance did X give on their Q3 call?")
2. Earnings performance question (e.g. "How did X's operating margin trend across 2023–2024 in their 10-Ks?")
3. Risk factor or 10-K question (e.g. "What key risk factors did X flag in their latest annual report?")
4. Segment or growth driver question (e.g. "What drove X's data center revenue growth in Q2 2024?")

Recent headlines:
{headline_text}

Return ONLY a JSON array of exactly 4 strings, no markdown, no explanation.
Example format: ["Question 1", "Question 2", "Question 3", "Question 4"]"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        import json
        raw = resp.choices[0].message.content.strip()
        # Extract JSON array even if wrapped in markdown
        import re
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            questions = json.loads(match.group())
            if isinstance(questions, list) and len(questions) >= 4:
                return [str(q) for q in questions[:4]]
    except Exception:
        pass

    return FALLBACK_QUESTIONS


@st.cache_data(ttl=1800, show_spinner=False)
def get_trending_questions():
    """
    Returns 4 trending example questions based on real-time news.
    Cached for 1 hour. Falls back to defaults on any error.
    """
    try:
        headlines = _fetch_headlines(max_per_ticker=2)
        if not headlines:
            return FALLBACK_QUESTIONS
        return _generate_questions(headlines)
    except Exception:
        return FALLBACK_QUESTIONS
