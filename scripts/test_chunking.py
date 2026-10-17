from pathlib import Path

DATA_DIR = Path("asia_documents")
CHUNK_SIZE = 25
OVERLAP = 5


def load_documents(directory: Path) -> list[dict[str, str]]:
    documents = []

    for path in sorted(directory.rglob("*.txt")):
        text = path.read_text(encoding="utf-8-sig").strip()

        if text:
            documents.append({
                "source": str(path),
                "text": text,
            })

    return documents


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    words = text.split()
    step = chunk_size - overlap

    return [
        " ".join(words[start : start + chunk_size])
        for start in range(0, len(words), step)
    ]


def main() -> None:
    documents = load_documents(DATA_DIR)

    if not documents:
        raise RuntimeError(f"No text documents found in {DATA_DIR}")

    chunks = []

    for document in documents:
        for index, text in enumerate(chunk_text(document["text"])):
            chunks.append({
                "source": document["source"],
                "chunk_index": index,
                "text": text,
            })

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunk size: {CHUNK_SIZE} words")
    print(f"Overlap: {OVERLAP} words")
    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks[:5]:
        print(
            f"\nSource: {chunk['source']}"
            f"\nChunk index: {chunk['chunk_index']}"
            f"\nWord count: {len(chunk['text'].split())}"
            f"\n{chunk['text']}"
        )


if __name__ == "__main__":
    main()
