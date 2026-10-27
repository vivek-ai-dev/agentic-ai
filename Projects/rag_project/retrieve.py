import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_PATH = "./chroma_db"

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name="company_policies"
)


def retrieve(
    query,
    top_k=3,
    category=None
):

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    where_filter = None

    print(f"query = {query}")
    print(f"top_k = {top_k}")
    if category:
        print(f"category = {category}")
        where_filter = {
            "category": category
        }

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter
    )

    return results

if __name__ == "__main__":

    results = retrieve(
        "How many annual leave days do employees receive?"
    )

    print(f"results = {results}")
    #print(results["documents"])