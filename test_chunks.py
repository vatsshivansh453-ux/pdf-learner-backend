import pickle


with open(
    "vector_store/chunks.pkl",
    "rb"
) as f:

    chunks = pickle.load(f)


for chunk in chunks[:3]:

    print(chunk)