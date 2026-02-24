import chromadb
from config import CHROMA_DIR  # Changed from CHROMA_SETTINGS
from embeddings import embed_query

def load_vector_store():
    
    # Use PersistentClient with the path directly
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection("ml_notes")

def retrieve_top_k(query, mode):
    collection = load_vector_store()

    #------- Query Conditioning -------------
    if mode == "summary":
        search_query = f"Detailed explanation only about {query}"
        k = 3

    if mode == "quiz":
        search_query = f"Important key concepts about {query}"
        k = 6

    else:
        search_query = query
        k = 4

    query_embedding = embed_query(search_query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    # Returns the list of document strings
    documents = results["documents"][0]

    """#---------- Post Filtering for Topic Purity ---------
    filtered = [
        doc for doc in documents
        if query.lower() in doc.lower()
    ]

    if filtered:
        return filtered
        
    return documents"""
    
    # Rank by keyword frequency
    scored_docs = sorted(
        documents,
        key= lambda doc: doc.lower().count(query.lower()),
        reverse= True
    )
    
    return scored_docs[:k]