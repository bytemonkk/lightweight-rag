
import requests
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
TOP_K = 3

DOCUMENTS = [
    "Paris is the capital of France and a major European city.",
    "Tokyo is the capital of Japan, known for its technology and culture.",
    "The Maldives is famous for tropical islands, beaches, and coral reefs.",
    "Cricket is a popular sport played with a bat and ball.",
    "Bali is an Indonesian island known for beaches and tourism.",
]

def retrieve_context(question: str, model: SentenceTransformer) -> list[str]:
    document_embeddings = model.encode(
        DOCUMENTS,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    retriever = NearestNeighbors(
        n_neighbors=TOP_K,
        metric="cosine",
        algorithm="brute",
    )
    retriever.fit(document_embeddings)

    _, indices = retriever.kneighbors(query_embedding)
    return [DOCUMENTS[index] for index in indices[0]]

def generate_answer(question: str, context: list[str]) -> str:
    prompt = f"""Answer the question using only the supplied context.
If the context does not contain the answer, say you do not have enough information.

Context:
{chr(10).join(context)}

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
    question = "Which places are famous for islands and beaches?"

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    context = retrieve_context(question, model)

    print("\nRetrieved context:")
    for index, passage in enumerate(context, start=1):
        print(f"{index}. {passage}")

    print("\nGenerating answer with Ollama...")
    answer = generate_answer(question, context)

    print("\nFinal answer:")
    print(answer)

if __name__ == "__main__":
    main()