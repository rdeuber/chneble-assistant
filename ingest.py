import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768
COLLECTION_NAME = "knowledge_chunks"

RAW_DIR = BASE_DIR / "data" / "raw"
NORMALIZED_DIR = BASE_DIR / "data" / "normalized"

PDF_FILE = RAW_DIR / "reglement.pdf"
EVENT_FILE = NORMALIZED_DIR / "event.json"
PRICES_FILE = NORMALIZED_DIR / "prices.csv"
RANKINGS_FILE = NORMALIZED_DIR / "rankings.csv"
SPONSORS_FILE = NORMALIZED_DIR / "sponsors.csv"


genai_client = genai.Client(
    http_options=types.HttpOptions(api_version="v1")
)

firestore_client = firestore.Client(project=PROJECT_ID)
collection = firestore_client.collection(COLLECTION_NAME)


def clean_text(text: str) -> str:
    """Replace repeated whitespace with single spaces."""
    return re.sub(r"\s+", " ", str(text)).strip()


def split_into_chunks(
    text: str,
    max_characters: int = 1_200,
    overlap: int = 200,
) -> list[str]:
    """Split text into overlapping chunks."""
    text = clean_text(text)

    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + max_characters, len(text))

        if end < len(text):
            word_boundary = text.rfind(
                " ",
                start + max_characters // 2,
                end,
            )

            if word_boundary > start:
                end = word_boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def load_pdf_chunks() -> list[dict]:
    """Extract page-aware chunks from the regulations PDF."""
    if not PDF_FILE.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_FILE}")

    reader = PdfReader(str(PDF_FILE))
    chunks: list[dict] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        for chunk_number, text in enumerate(
            split_into_chunks(page_text),
            start=1,
        ):
            chunks.append(
                {
                    "text": text,
                    "title": f"Reglement Chneblä – Seite {page_number}",
                    "source": "Reglement Chneblä",
                    "filename": PDF_FILE.name,
                    "page": page_number,
                    "row_number": None,
                    "chunk_number": chunk_number,
                    "source_type": "regulations",
                }
            )

    return chunks


def flatten_json(
    value: Any,
    prefix: str = "",
) -> list[tuple[str, str]]:
    """Flatten nested JSON into readable key-value pairs."""
    items: list[tuple[str, str]] = []

    if isinstance(value, dict):
        for key, child_value in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(flatten_json(child_value, child_prefix))

    elif isinstance(value, list):
        for index, child_value in enumerate(value, start=1):
            child_prefix = f"{prefix}[{index}]"
            items.extend(flatten_json(child_value, child_prefix))

    elif value is not None:
        items.append((prefix, clean_text(value)))

    return items


def load_event_chunks() -> list[dict]:
    """Load event information from normalized JSON."""
    if not EVENT_FILE.exists():
        raise FileNotFoundError(f"Event file not found: {EVENT_FILE}")

    with EVENT_FILE.open(encoding="utf-8") as file:
        event_data = json.load(file)

    flattened_items = flatten_json(event_data)

    text_parts = ["Event information for the Chneblä competition."]

    for key, value in flattened_items:
        if value:
            text_parts.append(f"{key}: {value}.")

    event_text = " ".join(text_parts)
    chunks: list[dict] = []

    for chunk_number, text in enumerate(
        split_into_chunks(event_text),
        start=1,
    ):
        chunks.append(
            {
                "text": text,
                "title": "Current event information",
                "source": "Event information",
                "filename": EVENT_FILE.name,
                "page": None,
                "row_number": None,
                "chunk_number": chunk_number,
                "source_type": "event_information",
            }
        )

    return chunks


def detect_csv_dialect(file_path: Path) -> csv.Dialect:
    """Detect whether a CSV uses commas, semicolons, or tabs."""
    with file_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        sample = file.read(4096)

    try:
        return csv.Sniffer().sniff(
            sample,
            delimiters=",;\t",
        )
    except csv.Error:
        return csv.excel


def row_to_text(
    row: dict[str, str],
    source_name: str,
) -> str:
    """Convert one CSV row into retrieval-friendly text."""
    fields: list[str] = []

    for key, value in row.items():
        clean_key = clean_text(key or "")
        clean_value = clean_text(value or "")

        if clean_key and clean_value:
            fields.append(f"{clean_key}: {clean_value}")

    if not fields:
        return ""

    return f"{source_name}. " + ". ".join(fields) + "."


def load_csv_chunks(
    file_path: Path,
    source_name: str,
    source_type: str,
) -> list[dict]:
    """Convert every non-empty CSV row into one document."""
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    dialect = detect_csv_dialect(file_path)
    chunks: list[dict] = []

    with file_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file, dialect=dialect)

        if not reader.fieldnames:
            raise RuntimeError(
                f"CSV has no header row: {file_path}"
            )

        for row_number, row in enumerate(reader, start=2):
            text = row_to_text(row, source_name)

            if not text:
                continue

            chunks.append(
                {
                    "text": text,
                    "title": source_name,
                    "source": source_name,
                    "filename": file_path.name,
                    "page": None,
                    "row_number": row_number,
                    "chunk_number": 1,
                    "source_type": source_type,
                }
            )

    return chunks


def load_all_chunks() -> list[dict]:
    """Load all raw and normalized knowledge sources."""
    return (
        load_pdf_chunks()
        + load_event_chunks()
        + load_csv_chunks(
            PRICES_FILE,
            source_name="Food and drink prices",
            source_type="prices",
        )
        + load_csv_chunks(
            RANKINGS_FILE,
            source_name="Historical rankings",
            source_type="rankings",
        )
        + load_csv_chunks(
            SPONSORS_FILE,
            source_name="Sponsors",
            source_type="sponsors",
        )
    )


def create_document_embedding(
    text: str,
    title: str,
) -> list[float]:
    """Create an embedding optimized for document retrieval."""
    response = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            title=title,
            output_dimensionality=EMBEDDING_DIMENSION,
        ),
    )

    if not response.embeddings:
        raise RuntimeError("Gemini returned no embedding.")

    values = response.embeddings[0].values

    if not values:
        raise RuntimeError(
            "Gemini returned an empty embedding."
        )

    if len(values) != EMBEDDING_DIMENSION:
        raise RuntimeError(
            "Unexpected embedding dimension: "
            f"expected {EMBEDDING_DIMENSION}, got {len(values)}"
        )

    return values


def clear_collection() -> None:
    """Delete previously ingested chunks for this prototype."""
    deleted_count = 0

    for document in collection.stream():
        document.reference.delete()
        deleted_count += 1

    print(f"Deleted {deleted_count} existing chunks.")


def create_document_id(chunk: dict) -> str:
    """Create a stable Firestore document ID."""
    identity = "|".join(
        [
            str(chunk["filename"]),
            str(chunk["page"]),
            str(chunk["row_number"]),
            str(chunk["chunk_number"]),
        ]
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:24]


def print_chunk_summary(chunks: list[dict]) -> None:
    """Print the number of chunks from each source."""
    counts: dict[str, int] = {}

    for chunk in chunks:
        source_type = chunk["source_type"]
        counts[source_type] = counts.get(source_type, 0) + 1

    print(f"Extracted {len(chunks)} chunks:")

    for source_type, count in sorted(counts.items()):
        print(f"  {source_type}: {count}")


def ingest() -> None:
    chunks = load_all_chunks()

    if not chunks:
        raise RuntimeError("No text chunks were extracted.")

    print_chunk_summary(chunks)
    clear_collection()

    for index, chunk in enumerate(chunks, start=1):
        embedding = create_document_embedding(
            text=chunk["text"],
            title=chunk["title"],
        )

        document_id = create_document_id(chunk)

        collection.document(document_id).set(
            {
                **chunk,
                "embedding": Vector(embedding),
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "ingested_at": firestore.SERVER_TIMESTAMP,
            }
        )

        location = (
            f"page {chunk['page']}"
            if chunk["page"] is not None
            else f"row {chunk['row_number']}"
            if chunk["row_number"] is not None
            else f"chunk {chunk['chunk_number']}"
        )

        print(
            f"[{index}/{len(chunks)}] "
            f"Stored {chunk['source']} ({location})"
        )

    print("Ingestion completed successfully.")


if __name__ == "__main__":
    ingest()