from openai import OpenAI
import os

# Initialize OpenAI client (handled within the module)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_PROMPT_MODEL = os.getenv("OPENAI_PROMPT_MODEL", "gpt-4")

client = OpenAI(api_key=OPENAI_API_KEY)

def ask_openai(question, context):
    """
    Call OpenAI API with the question and semantic search results as context.
    """
    # Construct the prompt with context
    prompt = f"""
    You are an expert assistant. Use the provided context to answer the user's question. 
    If the context does not contain sufficient information, explain this clearly to the user.

    Context:
    {context}

    Question:
    {question}

    Please provide a concise and accurate response strictly based on the context above.
    """

    # Create a chat completion
    response = client.chat.completions.create(
        model=OPENAI_PROMPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that only answers based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=200
    )

    # Return the assistant's response
    return response.choices[0].message.content.strip()