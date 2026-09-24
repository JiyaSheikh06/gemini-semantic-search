from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Initialize Gemini Embedding Model
embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001', dimensions = 300)

# Knowledge Base
documents = [
    "Artificial Intelligence is a field of computer science.",
    "Machine Learning allows computers to learn from data.",
    "Python is a popular programming language.",
    "Django is a Python framework for web development.",
    "FastAPI is a modern Python framework for building APIs.",
    "Large Language Models can generate human-like text.",
    "SQL is used to manage and query databases.",
    "RAG combines document retrieval with language models."
]

# Convert documents into embeddings
doc_embeddings = embedding.embed_documents(documents)

print('Semantic Search \n')
print("Type 'exit' to stop.\n")
while True:
    query = input('Please Enter Your Question:\n')

    if query.lower() == 'exit':
        print('GoodBye!')
        break

    if len(query.strip()) < 2:
        print("Please enter a meaningful question.\n")
        continue

    if not query.strip():
        continue

    # Convert query into embedding
    query_embedding = embedding.embed_query(query)

    # Calculate similarity scores
    scores = cosine_similarity([query_embedding],doc_embeddings)[0]

    # Get highest similarity score
    index, score = sorted(list(enumerate(scores)),key=lambda x : x[1])[-1]

    print("\nMost Relevant Document:")
    print(documents[index])

    print("Similarity Score:", round(score, 4))
    print("-" * 50)


