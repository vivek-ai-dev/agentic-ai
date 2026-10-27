from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


documents = [
    # AI
    "Artificial intelligence enables machines to perform tasks that normally require human intelligence.",

    # Python
    "Python is widely used for machine learning, data science, automation, and artificial intelligence.",

    # Java
    "Java is widely used for enterprise backend applications and large-scale distributed systems.",

    # Databases
    "Relational databases store structured data using tables, rows, columns, and relationships.",

    # Cloud
    "Cloud computing provides scalable computing, storage, and networking resources over the internet.",

    # Agentic AI
    "AI agents can reason about goals, select tools, take actions, observe results, and adapt their behavior.",

    # Machine Learning
    "Machine learning allows systems to learn patterns from data and make predictions without explicitly programmed rules.",

    # Mumbai
    "Mumbai is a major metropolitan city in India and an important financial and commercial center.",

    # Finance
    "Financial institutions use data analysis and machine learning for fraud detection, risk management, and forecasting.",

    # Cybersecurity
    "Cybersecurity protects computer systems, networks, applications, and data from unauthorized access and attacks."

]


# -----------------------------
# 1. Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 2. Create document embeddings
# -----------------------------

embeddings = model.encode(documents)

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)


# -----------------------------
# 3. Normalize embeddings
# -----------------------------

faiss.normalize_L2(embeddings)


# -----------------------------
# 4. Create FAISS index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


# -----------------------------
# 5. Interactive search
# -----------------------------

while True:

    query = input("\nEnter your query (or 'exit' to quit): ")

    if query.lower() == "exit":
        break

    # Generate query embedding
    query_embedding = model.encode([query])

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # Normalize query
    faiss.normalize_L2(query_embedding)

    # Search
    k = 3

    scores, indices = index.search(
        query_embedding,
        k
    )

    print(f"scores = {scores}, indices = {indices}")
    print("\nTop 3 Results:")
    print("-" * 60)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        print(f"\nRank: {rank}")
        print(f"Similarity: {score:.4f}")
        print(f"Document: {documents[idx]}")