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

def semantic_search(query_embedding, index, paragraphs, k=5):
    """
    Perform semantic search using a FAISS index and a precomputed query embedding.
    """
    # Search the FAISS index
    distances, indices = index.search(query_embedding, k)

    # Retrieve paragraphs based on search results
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < 0 or idx >= len(paragraphs):  # Ensure the index is valid
            print(f"Invalid index returned by FAISS: {idx}")
            continue
        
        paragraph = paragraphs[idx]
        results.append((paragraph, distance))

    if not results:
        raise ValueError("No valid results found. Check FAISS index or query alignment.")
    
    return results