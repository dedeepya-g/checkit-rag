import re
import html
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import searcher
import answerer
import trending

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Checkit Analytics", layout="wide", page_icon="📊")

# ── Checkit brand theme CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.8rem; max-width: 1100px; }

    div[data-testid="stTextInput"] input {
        font-size: 1.05rem;
        padding: 0.65rem 1rem;
        border-radius: 8px;
        border: 1.5px solid #046bd2 !important;
        background: transparent;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #045cb4 !important;
        box-shadow: 0 0 0 3px rgba(4,107,210,0.15) !important;
    }

    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
        border-radius: 20px;
        font-size: 0.82rem;
        border: 1.5px solid #046bd2;
        color: #046bd2;
        background: transparent;
        padding: 0.35rem 0.75rem;
        white-space: normal;
        line-height: 1.3;
        transition: all 0.15s ease;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:hover {
        background: #046bd2;
        color: white;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        border-radius: 6px;
        font-size: 0.8rem;
        text-align: left;
        background: rgba(4,107,210,0.08);
        border: none;
        color: inherit;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        background: rgba(4,107,210,0.18);
    }

    .citation-chip {
        display: inline-block;
        background: rgba(4,107,210,0.12);
        border: 1px solid rgba(4,107,210,0.35);
        border-radius: 14px;
        padding: 3px 10px;
        font-size: 0.78rem;
        color: #046bd2;
        margin: 2px 3px 2px 0;
        font-weight: 500;
    }

    .chat-question {
        background: rgba(4,107,210,0.08);
        border-left: 3px solid #046bd2;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem;
        margin: 1rem 0 0.4rem 0;
        font-size: 0.95rem;
        font-weight: 500;
    }

    .conf-badge {
        display: inline-block;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .conf-high   { background: rgba(4,120,87,0.10);  border: 1px solid rgba(4,120,87,0.30);  color: #047857; }
    .conf-medium { background: rgba(245,158,11,0.10); border: 1px solid rgba(245,158,11,0.35); color: #b45309; }
    .conf-low    { background: rgba(239,68,68,0.10);  border: 1px solid rgba(239,68,68,0.30);  color: #dc2626; }

    span[data-baseweb="tag"] {
        background-color: rgba(4,107,210,0.15) !important;
        border-radius: 4px !important;
    }

    /* Answer body — consistent font */
    .answer-body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-size: 0.93rem;
        line-height: 1.7;
        color: inherit;
    }
    .answer-body p  { margin: 0.4rem 0; }
    .answer-body ul { padding-left: 1.4rem; margin: 0.3rem 0; }
    .answer-body li { margin: 0.25rem 0; }
    .answer-body strong { font-weight: 650; }

    .num-up   { color: #16a34a; font-weight: 650; }
    .num-down { color: #dc2626; font-weight: 650; }
</style>
""", unsafe_allow_html=True)


# ── Answer colorizer ──────────────────────────────────────────────────────────
_UP_RE = re.compile(
    r"(?<!\w)"
    r"(\+\s*\d[\d,]*(?:\.\d+)?%"                        # +94%
    r"|(?:up|grew?|surged?|jumped?|rose?|gained?|expanded?|increased?)"
    r"\s+(?:by\s+)?\d[\d,]*(?:\.\d+)?%"                 # up 94%, grew 30%
    r"|\d[\d,]*(?:\.\d+)?%\s+(?:YoY|QoQ|CAGR)\s*(?:growth|increase|jump|surge)?"
    r")",                                                # 94% YoY
    re.IGNORECASE,
)
_DOWN_RE = re.compile(
    r"(?<!\w)"
    r"(-\s*\d[\d,]*(?:\.\d+)?%"                         # -5%
    r"|(?:down|fell?|dropped?|declined?|decreased?|contracted?|compressed?)"
    r"\s+(?:by\s+)?\d[\d,]*(?:\.\d+)?%"                 # down 5%, fell 3%
    r"|\d[\d,]*(?:\.\d+)?%\s+(?:YoY|QoQ)\s*(?:decline|decrease|drop|compression)"
    r")",
    re.IGNORECASE,
)


def _colorize(text: str) -> str:
    """Wrap positive/negative financial figures in coloured spans."""
    text = _UP_RE.sub(r'<span class="num-up">\1</span>', text)
    text = _DOWN_RE.sub(r'<span class="num-down">\1</span>', text)
    return text

ALL_TICKERS = [
    "AAPL", "MSFT", "META", "NVDA", "TSLA",
    "JPM", "GS", "SBUX", "WMT", "MCD",
    "NKE", "DIS", "CRM", "GOOGL", "AMZN",
    "NFLX", "ORCL", "MS", "BAC", "V",
]

# ── Session state ─────────────────────────────────────────────────────────────
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "last_run_query" not in st.session_state:
    st.session_state.last_run_query = ""
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "select_all" not in st.session_state:
    st.session_state.select_all = True
# Conversation: list of {query, answer, citations, confidence, elapsed, chunks}
if "conversation" not in st.session_state:
    st.session_state.conversation = []
# LLM message history for follow-up context
if "llm_history" not in st.session_state:
    st.session_state.llm_history = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo — rounded, white bg trimmed, no cloudy border
    st.markdown("""
        <style>
        section[data-testid="stSidebar"] img {
            border-radius: 12px;
            background: #ffffff;
            padding: 6px;
            display: block;
            margin: 0 auto 4px auto;
            box-shadow: 0 1px 4px rgba(0,0,0,0.18);
        }
        </style>
    """, unsafe_allow_html=True)
    col_l, col_img, col_r = st.columns([1, 2, 1])
    with col_img:
        st.image("checkit_analytics_logo.jpeg", use_container_width=True)
    st.markdown(
        "<h2 style='text-align:center;margin:4px 0 0 0;font-size:1.15rem'>Checkit Analytics</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#64748b;font-size:0.8rem;margin:2px 0 0 0'>"
        "Earnings Intelligence</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # New Conversation button — always visible above filters
    if st.session_state.conversation:
        if st.button("+ New Conversation", use_container_width=True, key="btn_new_conv"):
            st.session_state.conversation = []
            st.session_state.llm_history = []
            st.session_state.last_run_query = ""
            st.session_state.query_input = ""
            st.rerun()
        st.markdown("")

    # ── Filters ───────────────────────────────────────────────────────────────
    st.markdown("**Filters**")

    # Company
    st.markdown("<span style='font-size:0.8rem;color:#64748b'>Companies</span>", unsafe_allow_html=True)
    col_all, col_clear = st.columns(2)
    with col_all:
        if st.button("Select All", use_container_width=True, key="btn_all"):
            st.session_state.select_all = True
            st.rerun()
    with col_clear:
        if st.button("Clear All", use_container_width=True, key="btn_clear"):
            st.session_state.select_all = False
            st.rerun()

    default_tickers = ALL_TICKERS if st.session_state.select_all else []
    selected_tickers = st.multiselect(
        label="tickers",
        options=ALL_TICKERS,
        default=default_tickers,
        label_visibility="collapsed",
        placeholder="Choose companies...",
    )

    st.markdown("<span style='font-size:0.8rem;color:#64748b'>Filing Type</span>", unsafe_allow_html=True)
    filing_type = st.selectbox(
        label="filing_type",
        options=["All", "8-K  (Earnings Press Releases)", "10-K  (Annual Reports)", "10-Q  (Quarterly Reports)"],
        index=0,
        label_visibility="collapsed",
    )
    filing_type_clean = filing_type.split("  ")[0] if "  " in filing_type else filing_type

    st.markdown("<span style='font-size:0.8rem;color:#64748b'>Date Range</span>", unsafe_allow_html=True)
    year_range = st.slider(
        label="year_range",
        min_value=2020,
        max_value=2026,
        value=(2020, 2026),
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("700+ documents · 20 companies · 2020–2026")

    # Recent queries
    if st.session_state.query_history:
        st.divider()
        st.markdown("**Recent Queries**")
        for past_q in reversed(st.session_state.query_history[-5:]):
            if st.button(past_q, key=f"hist_{past_q}", use_container_width=True):
                st.session_state.query_input = past_q
                st.rerun()


METRIC_FILTERS = {}  # reserved for future use

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Earnings Intelligence")
st.markdown("Ask anything about earnings calls, filings, and financial performance")
st.markdown("")

# Trending example questions (only show when no active conversation)
if not st.session_state.conversation:
    example_queries = trending.get_trending_questions()

    cols = st.columns(4)
    for i, example in enumerate(example_queries):
        with cols[i]:
            if st.button(example, use_container_width=True, key=f"ex_{i}"):
                st.session_state.query_input = example
                st.rerun()
    st.markdown("")

# ── Display conversation history ──────────────────────────────────────────────
for i, turn in enumerate(st.session_state.conversation):
    # Question bubble
    st.markdown(
        f'<div class="chat-question">🔍 &nbsp;{html.escape(turn["query"])}</div>',
        unsafe_allow_html=True,
    )

    # Confidence + timing
    conf = turn["confidence"]
    badge_map = {
        "high":   ("● High Confidence",   "conf-high"),
        "medium": ("● Medium Confidence", "conf-medium"),
        "low":    ("● Low Confidence",    "conf-low"),
    }
    badge_text, badge_cls = badge_map[conf]
    col_badge, col_time = st.columns([3, 1])
    with col_badge:
        st.markdown(
            f'<span class="conf-badge {badge_cls}">{badge_text}</span>',
            unsafe_allow_html=True,
        )
    with col_time:
        st.markdown(
            f"<div style='text-align:right;padding-top:4px;color:#64748b;font-size:0.82em'>"
            f"{turn['elapsed']:.1f}s &nbsp;·&nbsp; {turn['chunks_used']} chunks</div>",
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.markdown(
            f'<div class="answer-body">{_colorize(turn["answer"])}</div>',
            unsafe_allow_html=True,
        )

    # Citations
    if turn["citations"]:
        chips_html = "".join(
            f'<span class="citation-chip">{c["ticker"]} · {c["quarter"]} · {c["form_type"]}</span>'
            for c in turn["citations"]
        )
        st.markdown(
            f'<div style="margin-top:0.5rem">'
            f'<span style="font-size:0.82rem;color:#64748b;font-weight:500;">Sources &nbsp;</span>'
            f'{chips_html}</div>',
            unsafe_allow_html=True,
        )

    # Source chunks
    if turn.get("chunks"):
        with st.expander(f"View {len(turn['chunks'])} sources"):
            for j, c in enumerate(turn["chunks"]):
                score_color = "#047857" if c["score"] >= 0.6 else "#f59e0b" if c["score"] >= 0.35 else "#ef4444"
                st.markdown(
                    f"**{j+1}.** `{c['ticker']}` &nbsp;{c['quarter']} &nbsp;·&nbsp; {c['form_type']}"
                    f" &nbsp;·&nbsp; {c.get('date','')}"
                    f'&nbsp;&nbsp;<span style="color:{score_color};font-size:0.8rem;font-weight:600">{c["score"]:.3f}</span>',
                    unsafe_allow_html=True,
                )
                safe_text = html.escape(c["text"][:350])
                st.markdown(
                    f'<div style="border-left:3px solid #046bd2;padding-left:12px;margin:4px 0 8px 0;'
                    f'color:#475569;font-size:0.87rem;line-height:1.5">'
                    f'{safe_text}{"..." if len(c["text"]) > 350 else ""}</div>',
                    unsafe_allow_html=True,
                )
                if j < len(turn["chunks"]) - 1:
                    st.divider()

    st.markdown("")

# ── Search input ──────────────────────────────────────────────────────────────
placeholder = "Continue the conversation..." if st.session_state.conversation else "e.g. What was Apple's revenue in Q3 2024?"

st.text_input(
    label="Query",
    placeholder=placeholder,
    label_visibility="collapsed",
    key="query_input",
)

if not st.session_state.conversation:
    st.caption("Press **Enter** to search · Supports any company, quarter, metric, or comparison")

# ── Run search ────────────────────────────────────────────────────────────────
query_to_run = st.session_state.query_input.strip()

if query_to_run and query_to_run != st.session_state.last_run_query:
    st.session_state.last_run_query = query_to_run

    if query_to_run not in st.session_state.query_history:
        st.session_state.query_history.append(query_to_run)
    if len(st.session_state.query_history) > 20:
        st.session_state.query_history = st.session_state.query_history[-20:]

    # Use a sorted tuple so cache keys are order-independent
    ticker_filter = tuple(sorted(selected_tickers)) if set(selected_tickers) != set(ALL_TICKERS) else None
    form_filter = filing_type_clean if filing_type_clean != "All" else None

    # Build date filter from year range slider
    year_start, year_end = year_range
    date_start = f"{year_start}-01-01" if year_start > 2020 else None
    date_end = f"{year_end}-12-31" if year_end < 2026 else None

    t0 = time.time()
    with st.spinner(""):
        try:
            chunks = searcher.search_with_fallback(
                query_to_run,
                ticker_filter=ticker_filter,
                form_type_filter=form_filter,
                date_start=date_start,
                date_end=date_end,
            )
            result = answerer.answer(
                query_to_run,
                chunks,
                conversation_history=st.session_state.llm_history,
            )
        except Exception as e:
            st.error(f"Search error: {e}")
            st.stop()

    elapsed = time.time() - t0

    if not chunks:
        st.warning("No relevant documents found. Try rephrasing or broadening your filters.")
        st.stop()

    # Save turn to conversation
    st.session_state.conversation.append({
        "query": query_to_run,
        "answer": result["answer"],
        "citations": result["citations"],
        "confidence": result["confidence"],
        "chunks_used": result["chunks_used"],
        "elapsed": elapsed,
        "chunks": chunks,
    })

    # Update LLM history for follow-ups (store condensed — no context block, just Q&A)
    st.session_state.llm_history.append({"role": "user", "content": query_to_run})
    st.session_state.llm_history.append({"role": "assistant", "content": result["answer"]})

    st.rerun()
