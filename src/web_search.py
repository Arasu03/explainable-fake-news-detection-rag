"""Search the public web for evidence snippets.

DDGS is a metasearch library, so results may occasionally be unavailable or
change format. Web results are treated as candidate evidence, not as proof.
"""
from ddgs import DDGS
from src.ingest import Evidence, get_embedding_model

def search_web_evidence(query, top_k=4):
    """Return web-search snippets as Evidence objects.

    The embedding model estimates relevance for display/ranking. It does not
    establish whether a claim is true.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max(1, top_k)))
    except Exception:
        # Keep the local knowledge-base path working if web search is down.
        return []

    candidates = []
    for item in results:
        title = (item.get("title") or "").strip()
        url = (item.get("href") or item.get("url") or "").strip()
        body = (item.get("body") or item.get("description") or "").strip()
        if not body:
            continue

        source = f"{title or 'Web result'} | {url}" if url else (title or "Web result")
        candidates.append(Evidence(text=body, source=source, similarity=0.0))

    if not candidates:
        return []

    # Rank snippets by semantic similarity to the user's claim.
    model = get_embedding_model()
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    text_vecs = model.encode(
        [f"{item.source}. {item.text}" for item in candidates],
        normalize_embeddings=True
    )
    for item, vec in zip(candidates, text_vecs):
        item.similarity = max(0.0, min(1.0, float(query_vec @ vec)))

    candidates.sort(key=lambda item: item.similarity, reverse=True)
    return candidates[:top_k]
