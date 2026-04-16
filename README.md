# Checkit Analytics: Earnings Intelligence

A RAG (Retrieval-Augmented Generation) system for querying SEC filings and earnings reports across 20 major public companies. Ask natural language questions about revenue, margins, guidance, risk factors, and financial trends to get analyst-quality answers grounded in source documents.

---

## Overview

Checkit Analytics ingests 700+ SEC filings (8-K, 10-K, 10-Q) spanning 2020-2026 and makes them searchable through a conversational interface. The system retrieves the most relevant document chunks, reranks them, and generates structured financial analysis using a large language model.

---

## Features

- **Natural language Q&A** over earnings press releases, annual reports, and quarterly filings
- **Hybrid search**: semantic (dense) + BM25 (sparse) vector retrieval via Pinecone
- **BM25 reranking**: fuses semantic and keyword scores before passing context to the LLM
- **Section-aware chunking**: 10-K and 10-Q filings split by SEC section headers (MD&A, Risk Factors, Financial Statements) for more coherent retrieval
- **Adaptive model routing**: simple factual queries use a fast 8B model; complex analysis routes to a 70B model
- **Same-company fallback**: if the exact quarter is unavailable, automatically retrieves the nearest available period for the same company
- **Live trending questions**: generated from real-time financial headlines each session
- **Conversation memory**: follow-up questions resolved using prior context
- **Confidence scoring**: each answer tagged High / Medium / Low based on retrieval score

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Vector Database** | Pinecone (serverless, multilingual-e5-large embeddings) |
| **LLM Inference** | Groq (`llama-3.3-70b-versatile` and `llama-3.1-8b-instant`) |
| **AI Platform** | Claude (Anthropic) |
| **Reranking** | BM25 via `rank_bm25`, fused with semantic scores |
| **Embeddings** | Pinecone Inference API, `multilingual-e5-large` (1024-dim) |
| **Data Source** | SEC EDGAR (8-K, 10-K, 10-Q filings) |
| **Language** | Python 3.12 |

---

## Companies Covered

`AAPL` `MSFT` `META` `NVDA` `TSLA` `JPM` `GS` `SBUX` `WMT` `MCD`
`NKE` `DIS` `CRM` `GOOGL` `AMZN` `NFLX` `ORCL` `MS` `BAC` `V`

---

## Project Structure

```
checkit-rag/
├── app.py               # Streamlit UI: search, conversation, filters
├── searcher.py          # Pinecone hybrid search, BM25 reranking, fallback logic
├── answerer.py          # Groq LLM integration, adaptive model routing, context builder
├── chunker.py           # Document chunking: section-aware (10-K/10-Q) and sliding window
├── pipeline.py          # Ingestion pipeline: embed and upsert documents to Pinecone
├── trending.py          # Live trending questions from earnings headlines via Groq
├── edgar_downloader.py  # SEC EDGAR filing downloader
└── requirements.txt     # Python dependencies
```

---

## Prerequisites

- Python 3.10 or higher
- A [Pinecone](https://www.pinecone.io) account with an API key
- A [Groq](https://console.groq.com) account with an API key
- SEC filing `.txt` files downloaded via `edgar_downloader.py`

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/dedeepya-g/checkit-rag.git
cd checkit-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

Get your keys at [pinecone.io](https://www.pinecone.io) and [console.groq.com](https://console.groq.com).

### 5. Download SEC filings

```bash
python edgar_downloader.py
```

### 6. Ingest documents into Pinecone

Update `DATA_FOLDER` in `chunker.py` to point to your downloaded filings, then run:

```bash
python pipeline.py
```

### 7. Run the app

```bash
streamlit run app.py
```

---

## Usage

- Use the **sidebar filters** to narrow by company, filing type (8-K / 10-K / 10-Q), and date range
- Type any question in the search bar and press **Enter**
- Ask follow-up questions; the system maintains conversation context automatically
- Expand **View sources** under any answer to inspect the retrieved document chunks

---

## License

This project is for educational and research purposes.
