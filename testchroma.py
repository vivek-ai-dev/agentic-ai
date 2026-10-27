import chromadb

# Create persistent Chroma client
client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Create or retrieve collection
collection = client.get_or_create_collection(
    name="company_docs"
)

# Add documents
collection.upsert(
    ids=[
        "hr_001",
        "tech_001",
        "db_001"
    ],

    documents=[
        "Employees receive 20 days of annual leave.",
        "Python is widely used for artificial intelligence.",
        "PostgreSQL is a relational database."
    ],

    metadatas=[
        {
            "department": "HR",
            "year": 2026
        },
        {
            "department": "Technology",
            "year": 2026
        },
        {
            "department": "Technology",
            "year": 2026
        }
    ]
)

# Semantic search
results = collection.query(
    query_texts=[
        "What is the employee vacation policy?"
    ],
    n_results=2
)

print(f"results = {results}")

print(f"ChromaDB Version = {chromadb.__version__}")