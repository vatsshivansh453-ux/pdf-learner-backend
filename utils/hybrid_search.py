import re
from rank_bm25 import BM25Okapi


def _tokenize(text):
    # Strip punctuation before splitting, so a chunk written as
    # "Machine Learning (ML)" or "AI," still matches a plain query
    # token like "ml" or "ai" instead of "(ml)" / "ai,".
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_search(question, pdf_chunks, k=5, user_id=None):

    # Scope to this user's own chunks first (if given) so keyword ranking
    # isn't diluted or crowded out by other users' documents in the shared
    # store — same fix as the FAISS side in vector_store.search_faiss_scoped.
    scoped_chunks = (
        [c for c in pdf_chunks if c.get("user_id") == user_id]
        if user_id is not None else pdf_chunks
    )

    if not scoped_chunks:
        return []

    # Tokenize all chunks
    corpus = [
        _tokenize(chunk["text"])
        for chunk in scoped_chunks
    ]

    bm25 = BM25Okapi(corpus)

    query = _tokenize(question)

    scores = bm25.get_scores(query)

    ranked = sorted(
        zip(scores, scoped_chunks),
        key=lambda x: x[0],
        reverse=True
    )

    return [
        chunk
        for score, chunk in ranked[:k]
    ]