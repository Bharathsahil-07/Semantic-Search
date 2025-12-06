import os, sys, subprocess

def ensure_package(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

ensure_package("pandas")
ensure_package("numpy")
ensure_package("sentence-transformers", "sentence_transformers")
ensure_package("rank-bm25", "rank_bm25")
ensure_package("scikit-learn", "sklearn")

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import re
from pathlib import Path

def simple_tokenize(text):
    return re.findall(r"\w+", str(text).lower())

def build_text_from_row(row):
    fields = [
        "book_id",
        "cover_image_uri",
        "book_title",
        "book_details",
        "format",
        "publication_info",
        "authorlink",
        "author",
        "num_pages",
        "genres",
    ]
    parts = []
    for f in fields:
        if f in row and pd.notna(row[f]):
            parts.append(str(row[f]))
    return " | ".join(parts)

def normalize_scores(arr):
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return arr
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val < 1e-9:
        return np.ones_like(arr)
    return (arr - min_val) / (max_val - min_val)

def main():
    csv_path = input("Enter path to CSV file: ").strip()
    if not csv_path:
        print("No CSV path given")
        return
    if not os.path.exists(csv_path):
        print("CSV file not found")
        return

    df = pd.read_csv(csv_path)
    corpus = df.apply(build_text_from_row, axis=1).tolist()

    base = Path(csv_path).stem
    emb_file = f"{base}_minilm_embeddings.npz"

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(emb_file):
        data = np.load(emb_file)
        embeddings = data["embeddings"]
        if embeddings.shape[0] != len(corpus):
            print("Existing embeddings size does not match CSV rows. Recomputing embeddings.")
            embeddings = model.encode(corpus, batch_size=32, show_progress_bar=True)
            np.savez(emb_file, embeddings=embeddings)
    else:
        print("No embeddings file found. Creating embeddings...")
        embeddings = model.encode(corpus, batch_size=32, show_progress_bar=True)
        np.savez(emb_file, embeddings=embeddings)

    tokenized_corpus = [simple_tokenize(text) for text in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    query = input("Enter your search query: ").strip()
    if not query:
        print("Empty query")
        return

    q_emb = model.encode([query])
    dense_scores = cosine_similarity(q_emb, embeddings)[0]
    q_tokens = simple_tokenize(query)
    bm25_scores = bm25.get_scores(q_tokens)

    dense_norm = normalize_scores(dense_scores)
    bm25_norm = normalize_scores(bm25_scores)
    alpha = 0.5
    hybrid_scores = alpha * bm25_norm + (1 - alpha) * dense_norm

    top_k = 5
    indices = np.argsort(-hybrid_scores)[:top_k]

    print(f"\nTop {top_k} results:\n")
    for rank, idx in enumerate(indices, start=1):
        row = df.iloc[idx]
        print(f"Result {rank} (score {hybrid_scores[idx]:.4f}):")
        print(f"  title: {row.get('book_title', '')}")
        print(f"  author: {row.get('author', '')}")
        print(f"  details: {row.get('book_details', '')}\n")

if __name__ == "__main__":
    main()
