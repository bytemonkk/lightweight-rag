# Lightweight Retrieval-Augmented Generation (RAG)

A lightweight Retrieval-Augmented Generation (RAG) pipeline built with Python, Sentence Transformers, scikit-learn, and Docker-hosted Ollama. The system retrieves relevant information from a collection of text documents and uses a local Large Language Model (LLM) to generate context-grounded answers.

## Overview

Large Language Models can generate answers using their pretrained knowledge, but that knowledge may not contain the information required for a specific document collection. RAG addresses this limitation by retrieving relevant documents and providing their contents to the LLM as context.

This project implements a lightweight RAG workflow using nine text documents about Asian countries. It demonstrates document loading, text chunking, embedding generation, semantic retrieval, and context-grounded answer generation.

### Key Features

- Loads and processes local text documents.
- Splits documents into overlapping word-based chunks.
- Generates semantic embeddings using Sentence Transformers.
- Retrieves relevant chunks using scikit-learn's `NearestNeighbors`.
- Generates answers using Qwen2.5 3B through Docker-hosted Ollama.
- Displays retrieved source documents and cosine distances.
- Instructs the LLM to abstain when the provided context is insufficient.
- Runs locally without requiring a cloud-based LLM API.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Dependency and environment management | UV |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimensions | 384 |
| Similarity search | scikit-learn `NearestNeighbors` |
| Distance metric | Cosine distance |
| Text generation | Qwen2.5 3B |
| Local model serving | Ollama |
| Containerization | Docker |
| HTTP communication | Python Requests |

## System Architecture

```text
                 User Question
                       |
                       v
              Query Embedding
             Sentence Transformers
                       |
                       v
              Semantic Retrieval
          scikit-learn NearestNeighbors
                       |
                       v
              Top-3 Relevant Chunks
                       |
                       v
              Context Construction
                       |
                       v
               Ollama API Request
                       |
                       v
                Qwen2.5 3B
                       |
                       v
              Context-Grounded Answer
                       |
                       v
             Answer and Source Chunks
```

The document preparation and indexing process runs before question answering:

```text
Local Text Documents
         |
         v
    Document Loading
         |
         v
      Text Chunking
  25 Words / 5-Word Overlap
         |
         v
   Sentence Embeddings
    384-Dimensional Vectors
         |
         v
  Nearest-Neighbor Index
         |
         v
   Ready for Retrieval
```

## Dataset

The project uses nine text documents about Asian countries.

Dataset source: [Open Datasets — Asia Documents](https://github.com/gakudo-ai/open-datasets/blob/main/asia_documents.zip?raw=true)

The downloaded documents are stored in the `asia_documents/` directory.

### Chunking Configuration

| Parameter | Value |
|---|---|
| Number of source documents | 9 |
| Chunk size | 25 words |
| Overlap | 5 words |
| Chunk step | 20 words |
| Total chunks in the current dataset | 31 |

Each document is divided into overlapping chunks. The overlap helps preserve some context between neighboring chunks.

## Embedding Generation

The project uses the `sentence-transformers/all-MiniLM-L6-v2` model to convert each text chunk into a numerical vector.

Each vector contains 384 values representing semantic features of the corresponding text.

For the current dataset, the embedding matrix has the shape:

```text
(31, 384)
```

These vectors allow the retriever to compare the meaning of a user's question with the meaning of the stored text chunks.

## Semantic Retrieval

The project uses scikit-learn's `NearestNeighbors` with cosine distance.

```python
NearestNeighbors(
    n_neighbors=3,
    metric="cosine",
    algorithm="brute",
)
```

The retriever returns the three nearest document chunks for a question.

A smaller cosine distance indicates greater similarity between the vectors. The retrieved chunks are then combined into a context that is supplied to the language model.

The implementation includes a reusable `SklearnRetriever` class with separate methods for building the index and retrieving relevant chunks.

## Answer Generation with Ollama

The generation component uses Qwen2.5 3B through the Ollama HTTP API.

```text
Python RAG Pipeline
        |
        v
http://localhost:11434/api/generate
        |
        v
Docker Container: scientific-ollama
        |
        v
Ollama Model: qwen2.5:3b
        |
        v
Generated Answer
```

The prompt instructs the model to:

- Answer using the supplied context.
- Avoid introducing unsupported facts.
- State when the context does not contain enough information.

This encourages grounded responses, although it does not guarantee that hallucinations will never occur.

## Project Structure

```text
lightweight-rag/
|
├── asia_documents/
│   └── Country text documents
|
├── data/
│   └── countries.txt
|
├── scripts/
│   ├── test_chunking.py
│   ├── test_document_retrieval.py
│   ├── test_embeddings.py
│   ├── test_rag_pipeline.py
│   ├── test_retrieval.py
│   └── test_sentence_chunking.py
|
├── main.py
├── pyproject.toml
├── uv.lock
├── .python-version
├── README.md
└── .gitignore
```

The `test_*.py` files are executable experiments used to validate individual components. The primary end-to-end RAG implementation is in `scripts/test_rag_pipeline.py`.

## Prerequisites

Install the following:

- Python 3.12
- UV
- Docker Desktop or another working Docker installation
- Sufficient memory and disk space for the embedding model and Qwen2.5 3B

The project has been developed and tested on Windows using PowerShell.

## Installation and Setup

### 1. Clone the repository

If you are setting up the project on a new machine:

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd lightweight-rag
```

Replace the placeholder with the actual repository URL.

### 2. Install dependencies

From the project root:

```powershell
uv sync
```

UV creates or synchronizes the project's virtual environment using the dependency definitions and lockfile.

### 3. Start Ollama using Docker

If the existing container is already running, skip container creation.

Check its status:

```powershell
docker ps --filter "name=scientific-ollama"
```

If you need to create the container for the first time:

```powershell
docker run -d `
  --name scientific-ollama `
  -p 11434:11434 `
  ollama/ollama:latest
```

Pull the required model:

```powershell
docker exec -it scientific-ollama ollama pull qwen2.5:3b
```

Verify the model:

```powershell
docker exec -it scientific-ollama ollama list
```

The Python application communicates with Ollama through `http://localhost:11434`.

### 4. Verify the dataset

Ensure the country text files are available in:

```text
asia_documents/
```

The current implementation expects `.txt` documents in that directory.

## Running the Project

Run commands from the project root.

### Test document chunking

```powershell
uv run python scripts/test_chunking.py
```

This loads the documents and reports the chunking results.

### Test embedding generation

```powershell
uv run python scripts/test_embeddings.py
```

This generates embeddings for the dataset and displays their dimensions.

### Test semantic retrieval

```powershell
uv run python scripts/test_retrieval.py
```

This retrieves the most relevant chunks for a sample question.

### Run the complete RAG pipeline

```powershell
uv run python scripts/test_rag_pipeline.py
```

The pipeline:

1. Loads the documents.
2. Creates overlapping chunks.
3. Generates embeddings for the chunks.
4. Builds the scikit-learn retrieval index.
5. Embeds each question.
6. Retrieves the three nearest chunks.
7. Sends the question and retrieved context to Ollama.
8. Displays the generated answer and retrieved sources.

## Example

**Question**

> Which Asian countries are best known for islands and beaches?

**Example answer**

> The Philippines and Thailand are best known for their islands and beaches.

The answer is generated from the retrieved document context, which includes passages describing the Philippine archipelago and Thailand's beaches and islands.

### Testing Insufficient Context

The pipeline also tests a question such as:

> What is the population of the Moon?

Because the dataset does not provide this information, the model is instructed to say that the available context is insufficient.

An abstention response is a useful sanity check, but it does not prove that the system is entirely free of hallucinations.

## Current Persistence Behavior

**The current implementation does not use a persistent vector database or save a reusable retrieval index.**

The original text documents remain stored on disk. However, every execution of `scripts/test_rag_pipeline.py` currently regenerates the document embeddings and rebuilds the nearest-neighbor index in memory.

When the Python process exits, the in-memory chunk collection, embedding matrix, and fitted retrieval index are discarded.

The embedding model is normally cached locally by the model library, so it does not necessarily need to be downloaded again on every run.

Persistent embeddings and index storage could be added in a future iteration to reduce startup time. This is not currently implemented.

## Limitations

- The dataset is small and contains only nine country documents.
- Retrieval quality depends on the embedding model, chunk boundaries, and available text.
- The retriever returns the nearest chunks even when they are not sufficiently relevant.
- The LLM can still misinterpret context or generate unsupported claims.
- There is no persistent vector index or database.
- Document updates require the pipeline to be rerun; there is no incremental indexing workflow.
- There is no web interface, authentication, or multi-user API.
- The current pipeline uses a fixed top-three retrieval setting.
- Evaluation is limited to a few manually tested questions.

## Future Improvements

Potential extensions include:

1. Persisting embeddings and the retrieval index between runs.
2. Adding incremental document ingestion and index updates.
3. Evaluating retrieval precision and recall on a larger test set.
4. Testing answer faithfulness and abstention behavior systematically.
5. Exploring metadata filtering and alternative retrieval strategies.
6. Exposing the pipeline through a FastAPI service or a user interface.

These are potential future improvements, not features currently implemented.

## Summary

This project demonstrates a lightweight, locally executable RAG pipeline using Sentence Transformers for semantic embeddings, scikit-learn for nearest-neighbor retrieval, and Docker-hosted Ollama for answer generation.

It provides a practical foundation for understanding how document retrieval and LLM generation can work together to produce context-grounded responses without introducing a full-scale vector database or orchestration framework.

The implementation is a baseline learning project; its retrieval quality, answer faithfulness, and performance should be evaluated further before using it for high-stakes applications.