from openai import OpenAI

with open("api_key.txt", "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

def embed_documents(chunks: list[str]) -> list[list[float]]:
    """
    Embeds a list of chunks in a single batch request.
    """
    if not chunks:
        return []

    # OpenAI allows up to 2048 input strings per request
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )

    # Extract all embeddings in order
    embeddings = [item.embedding for item in response.data]
    
    print(f"Successfully batched {len(embeddings)} embeddings.")
    return embeddings

def embed_query(query: str) -> list[float]:
    """
    Embed a single query for retrieval.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding