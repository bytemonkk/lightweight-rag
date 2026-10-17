import requests
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from test_chunking import DATA_DIR, load_documents, chunk_text

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
TOP_K = 3


def build_chunks() -> list[dict[str, str]]:
    """Load documents and split them into overlapping text chunks."""
    documents = load_documents(DATA_DIR)
    chunks = []

    for document in documents:
        for index, text in enumerate(chunk_text(document["text"])):
            chunks.append({
                "source": document["source"],
                "chunk_index": str(index),
                "text": text,
            })

    if not chunks:
        raise RuntimeError("No document chunks were generated.")

    return chunks


def retrieve_context(
    question: str,
    model: SentenceTransformer,
    chunks: list[dict[str, str]],
    document_embeddings,
) -> list[dict[str, str]]:
    """Retrieve the TOP_K chunks closest to the question embedding."""
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    retriever = NearestNeighbors(
        n_neighbors=min(TOP_K, len(chunks)),
        metric="cosine",
        algorithm="brute",
    )
    retriever.fit(document_embeddings)

    distances, indices = retriever.kneighbors(query_embedding)

    results = []

    for index, distance in zip(indices[0], distances[0]):
        chunk = chunks[int(index)].copy()
        chunk["distance"] = f"{float(distance):.4f}"
        results.append(chunk)

    return results


def generate_answer(
    question: str,
    context: list[dict[str, str]],
) -> str:
    """Generate an answer using the retrieved context and Ollama."""
    context_text = "\n\n".join(
        f"Source: {chunk['source']}\n{chunk['text']}"
        for chunk in context
    )

    prompt = f"""You are a question-answering assistant.

Answer the question using only the supplied context.
Do not introduce facts that are not supported by the context.
If the context does not contain enough information, say so.
Keep the answer concise and mention the relevant countries.

Context:
{context_text}

Question: {question}

Answer:"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=120,
    )
    response.raise_for_status()

    return response.json()["response"].strip()


def main() -> None:
    question = "Which Asian countries are known for their islands and beaches?"

    print(f"Loading documents from: {DATA_DIR}")
    chunks = build_chunks()

    print(f"Documents loaded: {len(load_documents(DATA_DIR))}")
    print(f"Total chunks: {len(chunks)}")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nGenerating document embeddings...")
    document_embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print(f"Embedding matrix shape: {document_embeddings.shape}")

    retrieved = retrieve_context(
        question,
        model,
        chunks,
        document_embeddings,
    )

    print(f"\nQuestion: {question}")
    print(f"\nTop {len(retrieved)} retrieved chunks:")

    for rank, chunk in enumerate(retrieved, start=1):
        print(f"\n{rank}. Cosine distance: {chunk['distance']}")
        print(f"Source: {chunk['source']}")
        print(chunk["text"])

    print(f"\nGenerating answer with Ollama ({OLLAMA_MODEL})...")
    answer = generate_answer(question, retrieved)

    print("\nFinal answer:")
    print(answer)


if __name__ == "__main__":
    main()
