import os
import numpy as np
from openai import OpenAI

def preprocess_text(text):
    """Normalize text for better matching."""
    return text.lower()

def get_openai_embeddings(paragraphs, client, batch_size=10, cache_file='embeddings.npy'):
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