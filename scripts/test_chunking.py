
from pathlib import Path

DATA_DIR = Path("data")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_documents(data_dir: Path) -> list[dict[str, str]]:
    documents = []

    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()

        if text:
            documents.append({
                "source": path.name,
                "text": text,
            })

    return documents


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks = []

    for start in range(0, len(text), chunk_size - overlap):
        chunk = text[start : start + chunk_size]

        if chunk:
            chunks.append(chunk)

    return chunks


def main() -> None:
    documents = load_documents(DATA_DIR)

    if not documents:
        raise RuntimeError(f"No non-empty .txt files found in {DATA_DIR}")

    all_chunks = []

    for document in documents:
        chunks = split_text(document["text"])

        for index, chunk in enumerate(chunks):
            all_chunks.append({
                "source": document["source"],
                "chunk_index": index,
                "text": chunk,
            })

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(all_chunks)}")

    for chunk in all_chunks:
        print(
            f"\nSource: {chunk['source']} | "
            f"Chunk: {chunk['chunk_index']} | "
            f"Characters: {len(chunk['text'])}"
        )
        print(chunk["text"])


if __name__ == "__main__":
    main()
