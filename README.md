# Chunk, Embed, and Prompt CLI

## Activate venv

```python3 -m venv venv```

```source venv/bin/activate```

```pip install -r requirements.txt```

## Run App

```python -m app```

## Data

Wikipedia topics are leveraged to pull text for RAG. Stored in `data/wikipedia_output.txt`, this can be manually edited for further testing. If the information is not present or clearly defined in the text it will not return in the semantic search for the LLM to answer the prompt.

Manually edit the `data/wikipedia_output.txt` and be sure to delete the `embeddings.npy` to re-embed the data.

## Questions

1. Knowledge Representation Questions:
- What is knowledge representation in artificial intelligence?
- What are some applications of knowledge representation in AI?
- What is an ontology, and how is it used in knowledge representation?
- Why is commonsense knowledge a challenging aspect of knowledge representation?
- What is default reasoning in the context of knowledge representation?
2. Python-Specific Questions:
- What are the key characteristics of Python’s syntax and semantics?
- How does Python handle indentation and block structure?
- What is the “walrus operator” (:=), and when was it introduced?
- How does Python differentiate between lists and tuples?
- What is a generator expression, and how does it differ from a list comprehension?
3. AI and Machine Learning Questions:
- What are the different types of machine learning mentioned in the content?
- What is the role of Bayesian networks in AI?
- How does reinforcement learning differ from supervised learning?
4. General Technology and Applications:
- What are some common uses of Python in scientific computing?
- How is Python used in natural language processing?
- What are the benefits of deep learning mentioned in the content?
5. Historical and Development Questions:
- What was Python named after, and why?
- What are Python Enhancement Proposals (PEPs), and why are they important?
- How does CPython differ from other Python implementations like PyPy or Stackless Python?