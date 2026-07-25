from fastembed import TextEmbedding


# Loaded once, shared by both single-text and batch embedding calls below.
#
# Uses fastembed (ONNX Runtime) instead of sentence-transformers (PyTorch).
# PyTorch alone typically uses 300-500MB of RAM just being imported, which
# on a memory-capped host like Render's free 512MB tier is often enough by
# itself to get the whole process OOM-killed before it ever starts serving
# requests. fastembed produces equivalent embeddings without that overhead.
#
# BAAI/bge-small-en-v1.5 outputs 384-dimensional vectors, same as the
# previous all-MiniLM-L6-v2 model, so nothing downstream (FAISS index
# dimension, etc.) needs to change.
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def create_embedding(text):
    """Embed a single piece of text (e.g. a user's question)."""

    embedding = list(model.embed([text]))[0]

    return embedding


def create_embeddings(chunks):
    """Embed a batch of texts at once (e.g. all chunks from an uploaded PDF)."""

    embeddings = list(model.embed(chunks))

    return embeddings