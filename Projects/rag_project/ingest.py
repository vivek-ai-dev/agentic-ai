import os
import chromadb
from sentence_transformers import SentenceTransformer


DOCUMENTS_DIR = "documents"
CHROMA_PATH = "./chroma_db"

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_or_create_collection(
    name="company_policies"
)


def load_documents():
    documents = []

    for filename in os.listdir(DOCUMENTS_DIR):

        if not filename.endswith(".txt"):
            continue

        path = os.path.join(DOCUMENTS_DIR, filename)

        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        documents.append({
            "filename": filename,
            "text": text
        })

    return documents


def chunk_text(text, chunk_size=500, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

def get_category(filename):

    if "leave" in filename:
        return "HR"

    if "insurance" in filename:
        return "HR"

    if "remote" in filename:
        return "Company Policy"

    return "General"

def ingest():

    documents = load_documents()

    ids = []
    texts = []
    embeddings = []
    metadatas = []

    counter = 1

    for document in documents:

        chunks = chunk_text(document["text"])

        for chunk in chunks:

            ids.append(f"chunk_{counter}")

            texts.append(chunk)

            embedding = embedding_model.encode(
                chunk
            ).tolist()

            embeddings.append(embedding)

            metadatas.append({
                "document": document["filename"],
                "year": 2026,
                "category": get_category(document["filename"])
            })

            counter += 1

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )


    print(f"collection.count() = {collection.count()}")
    print(f"Inserted {len(ids)} chunks.")
    print(f"metadatas = {metadatas}")


if __name__ == "__main__":
    ingest()