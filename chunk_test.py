from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Artificial Intelligence is a branch of computer science.
Machine Learning is a subset of AI.
Deep Learning uses neural networks.
AI is used in healthcare and education.
"""

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks=splitter.split_text(text)

for i , chunk in enumerate(chunks):
    print("CHUNK ",i+1)
    print(chunk)
    print("---------------------------------")