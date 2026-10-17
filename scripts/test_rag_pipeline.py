import requests
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from test_chunking import DATA_DIR, load_documents, chunk_text

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
TOP_K = 3


class SklearnRetriever:
    """Retrieve relevant document chunks using cosine distance."""

    def __init__(self, n_neighbors: int = TOP_K):
        self.retriever = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="brute",
        )
        self.chunks = []
        self.is_fitted = False

    def fit(self, chunks: list[dict], embeddings) -> None:
        """Build the index once using the document embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have equal lengths.")

        if not chunks:
            raise ValueError("Cannot build an index without document chunks.")

        self.chunks = chunks
        self.retriever.fit(embeddings)
        self.is_fitted = True

        print("Retriever index built successfully!")

    def retrieve(self, query_embedding) -> list[dict]:
        """Return the nearest chunks, ordered by cosine distance."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before retrieve().")

        distances, indices = self.retriever.kneighbors(
            [query_embedding],
            n_neighbors=min(TOP_K, len(self.chunks)),
        )

        results = []

        for index, distance in zip(indices[0], distances[0]):
            chunk = self.chunks[int(index)].copy()
            chunk["distance"] = float(distance)
            results.append(chunk)

        return results


def generate_rag_answer(question: str, context: list[dict]) -> str:
    """Generate an answer from retrieved context using Ollama."""
    context_text = "\n\n".join(
        f"Source: {chunk['source']}\n{chunk['text']}"
        for chunk in context
    )

    prompt = f"""Answer the question using only the supplied context.
Do not introduce unsupported facts.
If the context does not contain enough information, say so.

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



def ask_rag_pipeline(
    question: str,
    embedding_model: SentenceTransformer,
    retriever: SklearnRetriever,
) -> str:
    """Retrieve relevant chunks and generate a RAG answer."""
    print(f"Question: {question}")

    # 1. Embed the query
    query_vector = embedding_model.encode(question)

    # 2. Retrieve context using scikit-learn
    relevant_chunks = retriever.retrieve(query_vector)

    # 3. Generate the answer
    answer = generate_rag_answer(question, relevant_chunks)

    print(f"\nRAG Answer: {answer}")
    print("\n--- Sources used ---")

    for index, chunk in enumerate(relevant_chunks, start=1):
        print(f"{index}. {chunk['text'][:100]}...")

    return answer


def main() -> None:
    # Load documents and create chunks
    documents = load_documents(DATA_DIR)
    all_chunks = []

    for document in documents:
        for index, text in enumerate(chunk_text(document["text"])):
            all_chunks.append({
                "source": document["source"],
                "chunk_index": index,
                "text": text,
            })

    print(f"Documents loaded: {len(documents)}")
    print(f"Total chunks: {len(all_chunks)}")

    # Load the embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # Embed documents and build the index once
    print("\nGenerating document embeddings...")
    document_embeddings = embedding_model.encode(
        [chunk["text"] for chunk in all_chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print(f"Embedding matrix shape: {document_embeddings.shape}")

    retriever = SklearnRetriever()
    retriever.fit(all_chunks, document_embeddings)

    # Run a question through the complete RAG pipeline
    question = "Which Asian countries are best known for islands and beaches?"

    print(f"\nGenerating answer with Ollama ({OLLAMA_MODEL})...")
    ask_rag_pipeline(
        "Which Asian countries are best known for islands and beaches?",
        embedding_model,
        retriever,
    )

    print("\n" + "=" * 60)

    ask_rag_pipeline(
        "What is the population of the Moon?",
        embedding_model,
        retriever,
    )
    
    print("\n" + "=" * 60)
    
    ask_rag_pipeline(
        "What is the capital of Japan?",
        embedding_model,
        retriever,
    )


if __name__ == "__main__":
    main()
