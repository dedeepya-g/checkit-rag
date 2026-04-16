import os
import time
import hashlib
from dotenv import load_dotenv
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi

import chunker


def _token_to_idx(token: str) -> int:
    """Stable integer index for a token — must match searcher.py."""
    return int(hashlib.md5(token.encode()).hexdigest()[:8], 16)

load_dotenv()

INDEX_NAME = "checkit-rag"
EMBED_MODEL = "multilingual-e5-large"  # Pinecone hosted inference model
BATCH_SIZE = 96  # Pinecone inference allows up to 96 texts per call
PINECONE_DIM = 1024  # multilingual-e5-large dimension


def embed_batch(pc, texts):
    """Embed a list of texts using Pinecone inference API."""
    result = pc.inference.embed(
        model=EMBED_MODEL,
        inputs=texts,
        parameters={"input_type": "passage", "truncate": "END"},
    )
    return [item["values"] for item in result]


def build_bm25_sparse(corpus_tokens, batch_tokens):
    """
    Fit BM25 on the full corpus and return per-document sparse vectors.
    Indices are stable integer hashes of token strings (matches searcher.py).
    Returns list of {"indices": [...], "values": [...]} dicts.
    """
    bm25 = BM25Okapi(corpus_tokens)
    sparse_vecs = []
    for tokens in batch_tokens:
        unique_tokens = list(set(tokens))
        # get_scores returns a per-document score for each query term;
        # we use it to get the BM25 weight of each token in this document
        # by treating this document's unique terms as the "query"
        scores = bm25.get_scores(unique_tokens)
        # Map each token to its BM25 weight using a stable integer index
        token_weights = {}
        for tok, score in zip(unique_tokens, scores):
            if score > 0:
                idx = _token_to_idx(tok)
                token_weights[idx] = float(score)
        sparse_vecs.append({
            "indices": list(token_weights.keys()),
            "values":  list(token_weights.values()),
        })
    return sparse_vecs


def main():
    print("=== Checkit RAG Pipeline ===\n", flush=True)

    # 1. Load chunks
    print("Loading and chunking documents...", flush=True)
    chunks = chunker.load_all_chunks()
    print(flush=True)

    # Pre-tokenise all chunks for BM25 corpus
    print("Building BM25 corpus...", flush=True)
    corpus_tokens = [c["text"].lower().split() for c in chunks]

    # 2. Connect to Pinecone
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set in .env")
    pc = Pinecone(api_key=api_key)
    print("Connected to Pinecone.", flush=True)

    # 3. Create index if needed (dotproduct supports hybrid sparse+dense)
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{INDEX_NAME}' (dim={PINECONE_DIM}, dotproduct, hybrid)...", flush=True)
        pc.create_index(
            name=INDEX_NAME,
            dimension=PINECONE_DIM,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print("  Waiting for index to be ready...", flush=True)
            time.sleep(2)
        print("Index created.\n", flush=True)
    else:
        print(f"Index '{INDEX_NAME}' already exists.\n", flush=True)

    index = pc.Index(INDEX_NAME)

    # 4. Upsert in batches
    print(f"Embedding and uploading {len(chunks):,} chunks to Pinecone...", flush=True)
    uploaded = 0
    skipped = 0

    batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]

    for batch_idx, batch in enumerate(tqdm(batches, desc="Upserting")):
        first_id = f"{batch[0]['ticker']}_{batch[0]['form_type']}_{batch[0]['date']}_{batch[0]['chunk_index']}"

        # Skip if already uploaded
        try:
            result = index.fetch(ids=[first_id])
            if first_id in result.vectors:
                skipped += len(batch)
                continue
        except Exception:
            pass

        texts = [c["text"] for c in batch]
        batch_tokens = [t.lower().split() for t in texts]

        # Dense embeddings
        try:
            embeddings = embed_batch(pc, texts)
        except Exception as e:
            print(f"\n  Embed error: {e}", flush=True)
            skipped += len(batch)
            continue

        # BM25 sparse vectors (fit on full corpus)
        global_offset = batch_idx * BATCH_SIZE
        batch_corpus = corpus_tokens[global_offset:global_offset + len(batch)]
        sparse_vecs = build_bm25_sparse(corpus_tokens, batch_corpus)

        vectors = []
        for c, emb, sparse in zip(batch, embeddings, sparse_vecs):
            vid = f"{c['ticker']}_{c['form_type']}_{c['date']}_{c['chunk_index']}"
            metadata = {
                "ticker": c["ticker"],
                "form_type": c["form_type"],
                "date": c["date"],
                "quarter": c["quarter"],
                "speaker": c["speaker"],
                "chunk_index": c["chunk_index"],
                "source_file": c["source_file"],
                "text": c["text"][:1000],
            }
            vectors.append({
                "id": vid,
                "values": emb,
                "sparse_values": sparse,
                "metadata": metadata,
            })

        try:
            index.upsert(vectors=vectors)
            uploaded += len(batch)
        except Exception as e:
            print(f"\n  Upsert error: {e}", flush=True)
            skipped += len(batch)

    print(f"\nDone. {uploaded * BATCH_SIZE:,} vectors uploaded, {skipped * BATCH_SIZE:,} skipped.", flush=True)
    stats = index.describe_index_stats()
    print(f"Index now contains {stats.total_vector_count:,} vectors total.", flush=True)


if __name__ == "__main__":
    main()
