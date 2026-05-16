import json
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_PATH = "fiqh_data/fiqh_qa.json"
INDEX_PATH = "faiss_index/fiqh.index"
CHUNKS_PATH = "faiss_index/chunks.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_fiqh_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_chunks(entries):
    chunks = []
    for entry in entries:
        quran = ", ".join(entry.get("quran_refs", []))
        hadith = ", ".join(entry.get("hadith_refs", []))
        text = (
            f"Category: {entry['category']}\n"
            f"Question: {entry['question']}\n"
            f"Answer: {entry['answer']}\n"
            f"Quran References: {quran}\n"
            f"Hadith References: {hadith}\n"
            f"Madhab: {entry.get('madhab', 'Hanafi')}\n"
            f"Source: {entry.get('source', '')}"
        )
        chunks.append({
            "text": text,
            "category": entry["category"],
            "question": entry["question"],
            "answer": entry["answer"],
            "quran_refs": entry.get("quran_refs", []),
            "hadith_refs": entry.get("hadith_refs", []),
            "madhab": entry.get("madhab", "Hanafi"),
            "source": entry.get("source", ""),
        })
    return chunks


def build_index(chunks, model):
    texts = [c["text"] for c in chunks]
    print(f"Encoding {len(texts)} entries...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def main():
    os.makedirs("faiss_index", exist_ok=True)
    print("Loading fiqh data...")
    entries = load_fiqh_data()
    print(f"Loaded {len(entries)} Q&A entries.")

    chunks = build_chunks(entries)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    index = build_index(chunks, model)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved FAISS index to {INDEX_PATH}")
    print(f"Saved {len(chunks)} chunks to {CHUNKS_PATH}")
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
