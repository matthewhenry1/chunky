import os
import numpy as np
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file

from app.utils import chunk_text, load_wikipedia
from app.services.embeddings_service import get_query_embedding, get_openai_embeddings
from app.services.faiss_service import create_faiss_index, semantic_search
from app.services.openai_service import ask_openai

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_and_preprocess_data():
    """
    Load and preprocess text data from Wikipedia.
    """
    topics = os.environ["WIKIPEDIA_TOPICS"].split(";")
    text = load_wikipedia(topics)
    paragraphs = chunk_text(text)
    return paragraphs


def generate_embeddings(paragraphs):
    """
    Generate and normalize embeddings for paragraphs.
    """
    print("Generating embeddings using OpenAI's text-embedding-ada-002...")
    embeddings = get_openai_embeddings(paragraphs)
    return np.array([embedding / np.linalg.norm(embedding) for embedding in embeddings])


def interactive_search_cli(index, paragraphs):
    """
    Run the Semantic Search CLI for user interaction.
    """
    print("\nSemantic Search CLI")
    print("Type 'exit' to quit the program.\n")

    while True:
        query = input("Enter your search query: ")
        if query.lower() == "exit":
            print("Exiting. Goodbye!")
            break

        # Generate the query embedding
        query_embedding = get_query_embedding(query)

        # Perform semantic search
        results = semantic_search(query_embedding, index, paragraphs, k=5)

        # Combine top matches into a context for OpenAI
        context = "\n".join([f"{i+1}. {paragraph.strip()}" for i, (paragraph, _) in enumerate(results)])

        print("\nTop Matches:")
        for i, (paragraph, distance) in enumerate(results, 1):
            print(f"{i}. {paragraph.strip()} (Score: {distance:.2f})\n")

        # Get response from OpenAI
        answer = ask_openai(query, context)
        print("\nOpenAI's Response:")
        print(answer)

def main():
    """
    Main entry point for the application.
    """
    # Load and preprocess data
    paragraphs = load_and_preprocess_data()

    # Generate embeddings
    embeddings = generate_embeddings(paragraphs)

    # Create FAISS index
    index = create_faiss_index(embeddings)
    print(f"FAISS index created with {len(paragraphs)} paragraphs.")

    # Start interactive CLI
    interactive_search_cli(index, paragraphs)


if __name__ == "__main__":
    main()