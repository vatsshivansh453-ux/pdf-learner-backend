import faiss
import numpy as np
import pickle
import os

print("VECTOR STORE IMPORTED")

def create_faiss_index(embedding):
    embedding=np.array(embedding).astype("float32")
    
    dimension=embedding.shape[1]
    
    index=faiss.IndexFlatL2(dimension)
    
    index.add(embedding)
    
    return index

def add_embeddings(index, embeddings):
    embeddings = np.array(embeddings).astype("float32")
    index.add(embeddings)

def search_faiss(index, query_embedding, chunks, k=2):

    query_embedding = np.array(
        [query_embedding]
    ).astype("float32")


    distances, indices = index.search(
        query_embedding,
        k
    )


    results = []


    for position, i in enumerate(indices[0]):

        results.append(
            {
                "file_name": chunks[i]["file_name"],

                "page_number": chunks[i]["page_number"],

                "chunk_number": chunks[i]["chunk_number"],

                "distance": float(distances[0][position]),

                "text": chunks[i]["text"],

                "user_id": chunks[i].get("user_id")
            }
        )


    return results


def search_faiss_scoped(index, query_embedding, chunks, user_id, k=15):
    """
    Same idea as search_faiss, but searches ONLY within a specific user's
    own chunks instead of the whole shared index.

    The FAISS index is shared across every user's uploads, so a plain
    index.search(k) call ranks a user's chunks against everyone else's —
    if the store has a lot of other content in it, a user's own best-matching
    chunk can get pushed out of the top-k before we ever get a chance to
    filter by user. Scoping BEFORE searching (by reconstructing just this
    user's vectors from the index and ranking only among those) guarantees
    the right chunk is always considered, no matter how big the shared
    store gets.
    """

    user_positions = [
        i for i, chunk in enumerate(chunks)
        if chunk.get("user_id") == user_id
    ]

    if not user_positions:
        return []

    # Pull the original vectors back out of the index for just this user's
    # chunks (IndexFlatL2 stores vectors verbatim, so reconstruct is exact).
    user_vectors = np.array(
        [index.reconstruct(i) for i in user_positions]
    ).astype("float32")

    query_vector = np.array(query_embedding).astype("float32")

    # L2 distance to every one of this user's vectors, ranked ascending
    # (same distance metric IndexFlatL2 uses internally).
    diffs = user_vectors - query_vector
    distances = np.einsum("ij,ij->i", diffs, diffs)

    order = np.argsort(distances)[:k]

    results = []

    for rank in order:
        i = user_positions[rank]
        chunk = chunks[i]

        results.append(
            {
                "file_name": chunk["file_name"],

                "page_number": chunk["page_number"],

                "chunk_number": chunk["chunk_number"],

                "distance": float(distances[rank]),

                "text": chunk["text"],

                "user_id": chunk.get("user_id")
            }
        )

    return results

def save_vector_store(index, chunks):
    print("Inside save_vector_store")

    os.makedirs("vector_store", exist_ok=True)

    faiss.write_index(index, "vector_store/faiss.index")
    print("FAISS index written")

    with open("vector_store/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Chunks written")
    print("Finished save_vector_store")
    
def load_vector_store():

    if not os.path.exists("vector_store/faiss.index"):
        return None, []


    index = faiss.read_index(
        "vector_store/faiss.index"
    )


    with open("vector_store/chunks.pkl","rb") as f:
        chunks = pickle.load(f)


    print("Vector store loaded")
    print("Chunks:",len(chunks))
    print("Vectors:",index.ntotal)


    return index, chunks

def rebuild_faiss_index(embeddings):
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index