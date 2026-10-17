
import re
from pathlib import Path

DATA_PATH = Path("data/countries.txt")
CHUNK_SIZE = 300
CHUNK_OVERLAP_SENTENCES = 1


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def chunk_sentences(
    sentences: list[str],
    chunk_size: int = CHUNK_SIZE,
    overlap_sentences: int = CHUNK_OVERLAP_SENTENCES,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap_sentences < 0:
        raise ValueError("overlap_sentences cannot be negative")

    chunks = []
    current_sentences = []
    current_length = 0

    for sentence in sentences:
        additional_length = len(sentence) + (1 if current_sentences else 0)

        if current_sentences and current_length + additional_length > chunk_size:
            chunks.append(" ".join(current_sentences))

            overlap = current_sentences[-overlap_sentences:] if overlap_sentences else []
            current_sentences = overlap
            current_length = len(" ".join(current_sentences))

        if current_sentences:
            current_length += 1

        current_sentences.append(sentence)
        current_length += len(sentence)

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)

    text = DATA_PATH.read_text(encoding="utf-8")
    sentences = split_into_sentences(text)
    chunks = chunk_sentences(sentences)

    print(f"Sentences detected: {len(sentences)}")
    print(f"Chunks created: {len(chunks)}")

    for index, chunk in enumerate(chunks):
        print(f"\nChunk {index}: {len(chunk)} characters")
        print(chunk)


if __name__ == "__main__":
    main()
