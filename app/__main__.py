import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from app.utils import chunk_text, load_text, load_wikipedia
from app.services.embeddings_service import preprocess_text, get_openai_embeddings
from app.services.faiss_service import create_faiss_index, semantic_search, boost_score
from app.services.openai_service import ask_openai

# Load environment variables from .env file
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Retrieve OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def main():
    # Load and preprocess text
    # file_path = "https://www.gutenberg.org/cache/epub/64317/pg64317.txt"
    # text = load_text(file_path)

    topics = ["Python (programming language)", "Artificial intelligence", "Machine learning"]
    text = load_wikipedia(topics)

    paragraphs = chunk_text(text)
    paragraphs = [preprocess_text(p) for p in paragraphs]

    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Generate or load embeddings
    print("Generating embeddings using OpenAI's text-embedding-ada-002...")
    embeddings = get_openai_embeddings(paragraphs, client)

    # Normalize embeddings for cosine similarity
    embeddings = np.array([embedding / np.linalg.norm(embedding) for embedding in embeddings])

    # Create FAISS index
    index = create_faiss_index(embeddings)
    print(f"FAISS index created with {len(paragraphs)} paragraphs.")

    print("\nSemantic Search CLI")
    print("Type 'exit' to quit the program.\n")

    while True:
        query = input("Enter your search query: ")
        if query.lower() == "exit":
            print("Exiting. Goodbye!")
            break

        # Perform semantic search
        results = semantic_search(query, index, paragraphs, client, boost_score, k=5)

        # Combine top matches into a context for OpenAI
        context = "\n".join([f"{i+1}. {paragraph.strip()}" for i, (paragraph, _) in enumerate(results)])

        print("\nTop Matches:")
        for i, (paragraph, distance) in enumerate(results, 1):
            print(f"{i}. {paragraph.strip()} (Score: {distance:.2f})\n")

        # Call OpenAI to get an answer
        answer = ask_openai(query, context, client)
        print("\nOpenAI's Response:")
        print(answer)

if __name__ == "__main__":
    main()