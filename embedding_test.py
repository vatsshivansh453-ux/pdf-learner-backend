from sentence_transformers import SentenceTransformer

model=SentenceTransformer(
    "all-MiniLm-L6-v2"
) 

text = "Artificial Intelligence is a branch of computer science"

embedding=model.encode(text)

print(embedding)
print(len(embedding))