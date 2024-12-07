import os
import numpy as np
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_SERVICE = os.getenv("EMBEDDING_SERVICE", "openai")  # Options: 'openai', 'langchain'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-ada-002")
LANGCHAIN_MODEL = os.getenv("LANGCHAIN_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Core Embedding Methods
def openai_generate_embeddings(input_text):
    """
    Generate embeddings using OpenAI's model.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        model=OPENAI_MODEL,
        input=input_text
    )
    if not response or not hasattr(response, 'data') or not response.data:
        raise ValueError("Invalid response from OpenAI embeddings API.")
    return [item.embedding for item in response.data]

def langchain_generate_embeddings(input_text, query=False):
    """
    Generate embeddings using LangChain's HuggingFaceEmbeddings model.
    """
    langchain_embeddings = HuggingFaceEmbeddings(model_name=LANGCHAIN_MODEL)
    if query:
        return langchain_embeddings.embed_query(input_text)
    return langchain_embeddings.embed_documents(input_text)

# Service Methods
def get_openai_document_embeddings(paragraphs, batch_size=10, cache_file='openai_embeddings.npy'):
    """
    Generate embeddings for paragraphs using OpenAI and handle file operations.
    """
    print(f"Generating embeddings using OpenAI's {OPENAI_MODEL}...")
    if os.path.exists(cache_file):
        print("Loading cached embeddings...")
        return np.load(cache_file)

    embeddings = []
    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i + batch_size]
        embeddings.extend(openai_generate_embeddings(batch))
    embeddings = np.array(embeddings)
    np.save(cache_file, embeddings)
    return embeddings

def get_langchain_document_embeddings(paragraphs, cache_file='langchain_embeddings.npy'):
    """
    Generate embeddings for paragraphs using LangChain and handle file operations.
    """
    print(f"Generating embeddings using LangChain's {LANGCHAIN_MODEL}...")
    if os.path.exists(cache_file):
        print("Loading cached embeddings...")
        return np.load(cache_file)
    
    embeddings = langchain_generate_embeddings(paragraphs)
    embeddings = np.array(embeddings)
    np.save(cache_file, embeddings)
    return embeddings

def get_query_embedding(query):
    """
    Generate embedding for a single query using the selected embedding service.
    """
    if EMBEDDING_SERVICE == "openai":
        embedding = openai_generate_embeddings([query])[0]
    elif EMBEDDING_SERVICE == "langchain":
        embedding = langchain_generate_embeddings(query, query=True)
    else:
        raise ValueError(f"Unsupported embedding service: {EMBEDDING_SERVICE}")
    return np.array(embedding, dtype=np.float32).reshape(1, -1)

def get_document_embedding(paragraphs):
    """
    Generate embeddings for a dataset (paragraphs) using the selected embedding service.
    """
    if EMBEDDING_SERVICE == "openai":
        return get_openai_document_embeddings(paragraphs)
    elif EMBEDDING_SERVICE == "langchain":
        return get_langchain_document_embeddings(paragraphs)
    else:
        raise ValueError(f"Unsupported embedding service: {EMBEDDING_SERVICE}")