def ask_openai(question, context, client):
    """
    Call OpenAI API with the question and semantic search results as context.
    """
    # Construct the prompt with context
    prompt = f"""
    You are an expert assistant. Use the following context from The Great Gatsby to answer the user's question.

    Context:
    {context}

    Question:
    {question}

    Please provide a concise and clear response.
    """

    # Create a chat completion
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant knowledgeable about literature."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=200
    )

    # Return the assistant's response
    return response.choices[0].message.content.strip()