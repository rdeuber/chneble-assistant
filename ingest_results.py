import csv
import hashlib
from pathlib import Path

from google.cloud import firestore


BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "data" / "normalized" / "rankings.csv"
COLLECTION_NAME = "results"


def create_document_id(row: dict[str, str]) -> str:
    """Create a stable ID so rerunning ingestion updates the same document."""
    identity = "|".join(
        [
            row["year"].strip(),
            row["category"].strip(),
            row["rank"].strip(),
            row["row_text"].strip(),
            row["source_pdf"].strip(),
        ]
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:24]


def ingest_results() -> None:
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

    firestore_client = firestore.Client()
    collection = firestore_client.collection(COLLECTION_NAME)

    required_columns = {
        "year",
        "category",
        "rank",
        "row_text",
        "source_pdf",
    }

    imported = 0
    batch = firestore_client.batch()
    batch_size = 0

    with CSV_FILE.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("The CSV file has no header.")

        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                f"Missing CSV columns: {sorted(missing_columns)}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                year = int(row["year"].strip())

                rank_text = row["rank"].strip()
                if not rank_text:
                    rank = None
                    rank_label = None
                else:
                    try:
                        rank = int(rank_text)
                        rank_label = None
                    except ValueError:
                        rank = None
                        rank_label = rank_text

                category = row["category"].strip()
                row_text = row["row_text"].strip()
                source_pdf = row["source_pdf"].strip()
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid data on CSV line {line_number}: {row}"
                ) from error

            if not category or not row_text or not source_pdf:
                raise ValueError(
                    f"Missing required value on CSV line {line_number}: {row}"
                )

            document = {
                "year": year,
                "category": category,
                "category_normalized": category.casefold(),
                "rank": rank,
                "row_text": row_text,
                "row_text_normalized": row_text.casefold(),
                "source_pdf": source_pdf,
                "ingested_at": firestore.SERVER_TIMESTAMP,
            }

            document_id = create_document_id(row)
            document_reference = collection.document(document_id)

            batch.set(document_reference, document)

            batch_size += 1
            imported += 1

            # Stay below Firestore's batch-write limit.
            if batch_size == 400:
                batch.commit()
                batch = firestore_client.batch()
                batch_size = 0

    if batch_size:
        batch.commit()

    print(f"Imported {imported} ranking records into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    ingest_results()
