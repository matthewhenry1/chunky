import os
import numpy as np
from openai import OpenAI

# Initialize OpenAI client within the module
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def get_openai_embeddings(paragraphs, batch_size=10, cache_file='embeddings.npy'):
    """
    Generate embeddings using OpenAI's text-embedding-ada-002 model.
    Caches embeddings locally to avoid redundant API calls.
    """
    if os.path.exists(cache_file):
        print("Loading cached embeddings...")
        return np.load(cache_file)

    embeddings = []
    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=batch
        )
        embeddings.extend([item.embedding for item in response.data])
    embeddings = np.array(embeddings)
    np.save(cache_file, embeddings)
    return embeddings

def get_query_embedding(query):
    """
    Generate embedding for a single query using OpenAI's text-embedding-ada-002 model.
    """
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    if not response or not hasattr(response, 'data') or not response.data:
        raise ValueError("Invalid response from embeddings API.")
    
    return np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)