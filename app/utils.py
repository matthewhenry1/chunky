import re
import requests
import wikipediaapi
import os

def load_text(file_path):
    """
    Load text content from a local file or a remote URL.
    """
    if file_path.startswith("http://") or file_path.startswith("https://"):
        # Fetch the text from the URL
        response = requests.get(file_path)
        response.raise_for_status()  # Raise an error for bad HTTP status
        return response.text
    else:
        # Load from a local file
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

def chunk_text(text, chunk_size=1000):
    """
    Chunk text into meaningful paragraphs.
    Handles paragraphs separated by blank lines and merges line breaks within paragraphs.
    """
    # Split the text into paragraphs based on blank lines
    paragraphs = re.split(r"\n\s*\n", text.strip())

    # Merge lines within a paragraph to create continuous text
    merged_paragraphs = [" ".join(p.splitlines()) for p in paragraphs]

    # Further split large paragraphs into smaller chunks
    chunks = []
    for paragraph in merged_paragraphs:
        if len(paragraph) > chunk_size:
            # Break large paragraphs into smaller chunks
            chunks.extend([paragraph[i:i+chunk_size] for i in range(0, len(paragraph), chunk_size)])
        else:
            chunks.append(paragraph)

    return chunks

def load_wikipedia(topics=None):
    """
    Load text content from one or more Wikipedia articles.
    If no topics are provided, defaults to a predefined topic list.
    """
    # Set a user-friendly User-Agent as required by Wikipedia's API policy
    user_agent = "chunky/1.0 (https://github.com/matthewhenry1/chunky; matthewanthonyhenry@gmail.com)"
    wiki = wikipediaapi.Wikipedia(language='en', user_agent=user_agent)

    if topics is None:
        topics = ["Python (programming language)", "Artificial intelligence", "Machine learning"]

    articles = []
    for topic in topics:
        page = wiki.page(topic)
        if page.exists():
            print(f"Fetching content for topic: {topic}")
            articles.append(page.text)
        else:
            print(f"Topic '{topic}' does not exist.")

    # Combine all articles into one large text
    result = "\n\n".join(articles)

    # Ensure the directory exists
    os.makedirs("data", exist_ok=True)
    with open("data/wikipedia_output.txt", "w", encoding="utf-8") as file:
        file.write(result)

    return result