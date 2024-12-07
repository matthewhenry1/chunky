import numpy as np
import faiss

def create_faiss_index(embeddings):
    """
    Create a FAISS index for efficient similarity search.
    """
    # Normalize embeddings for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def boost_score(paragraph, query):
    """Boost score for paragraphs containing exact query keywords."""
    keywords = query.split()
    return sum(1 for word in keywords if word in paragraph)

def semantic_search(query, index, paragraphs, client, boost_score=None, k=5):
    """
    Perform semantic search using a FAISS index.
    """
    # Generate embedding for the query
    response = client.embeddings.create(input=query, model="text-embedding-ada-002")
    if not response or not hasattr(response, 'data') or not response.data:
        raise ValueError("Invalid response from embeddings API.")
    
    # Extract the embedding and ensure it is a NumPy array
    query_embedding = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
    
    # Search the FAISS index
    distances, indices = index.search(query_embedding, k)

    # Retrieve paragraphs and optionally apply a boost score
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < 0 or idx >= len(paragraphs):  # Ensure the index is valid
            print(f"Invalid index returned by FAISS: {idx}")
            continue
        
        paragraph = paragraphs[idx]
        if boost_score:
            distance = boost_score(paragraph, query)  # Pass paragraph and query to the boost_score function
        results.append((paragraph, distance))

    if not results:
        raise ValueError("No valid results found. Check FAISS index or query alignment.")
    
    return results