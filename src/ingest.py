from dataclasses import dataclass
from pathlib import Path
import hashlib

import chromadb
from sentence_transformers import SentenceTransformer
import config

@dataclass
class Evidence:
    text: str
    source: str
    similarity: float

_embedding_model = None
_client = None
_collection = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedding_model

def get_collection():
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if _collection is None:
        _collection = _client.get_or_create_collection(
            name="trusted_evidence",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

def split_into_chunks(text, chunk_words=None, overlap=None):
    chunk_words = chunk_words or config.CHUNK_WORDS
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks

def build_index():
    config.KB_DIR.mkdir(parents=True, exist_ok=True)
    documents, ids, metadatas = [], [], []
    for path in sorted(list(config.KB_DIR.glob("*.txt")) + list(config.KB_DIR.glob("*.md"))):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for index, chunk in enumerate(split_into_chunks(text)):
            documents.append(chunk)
            ids.append(hashlib.sha256(f"{path.name}:{index}:{chunk}".encode()).hexdigest())
            metadatas.append({"source": path.name, "chunk": index})

    collection = get_collection()
    # Clear the old index so removed/changed files do not remain as stale evidence.
    existing = collection.get()
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])

    if not documents:
        return 0

    embeddings = get_embedding_model().encode(documents, normalize_embeddings=True).tolist()
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(documents)

def retrieve_evidence(query, top_k=4):
    collection = get_collection()
    if collection.count() == 0:
        return []
    embedding = get_embedding_model().encode([query], normalize_embeddings=True).tolist()[0]
    result = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    evidence = []
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for doc, meta, distance in zip(docs, metas, distances):
        # Chroma cosine distance is generally 0 (closest) to 2 (farthest).
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        evidence.append(Evidence(text=doc, source=meta.get("source", "unknown"), similarity=similarity))
    return evidence
