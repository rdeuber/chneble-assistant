from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from google.api_core.exceptions import NotFound
from google.cloud import firestore, storage


COLLECTION_NAME = "live_ranking"
DOCUMENT_ID = "current"
ZURICH_TIMEZONE = ZoneInfo("Europe/Zurich")


def get_status_reference(
    firestore_client: firestore.Client,
):
    return (
        firestore_client
        .collection(COLLECTION_NAME)
        .document(DOCUMENT_ID)
    )


def publish_pdf(
    pdf_path: Path,
    project_id: str,
    bucket_name: str,
) -> None:
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF does not exist: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, received: {pdf_path.name}"
        )

    now = datetime.now(ZURICH_TIMEZONE)

    # A unique name prevents browsers from showing an older cached PDF.
    object_name = (
        f"live/zwischenrangliste-"
        f"{now:%Y%m%d-%H%M%S}.pdf"
    )

    firestore_client = firestore.Client(
        project=project_id
    )
    status_reference = get_status_reference(
        firestore_client
    )

    current_status = status_reference.get()
    previous_object_name = None

    if current_status.exists:
        previous_object_name = (
            current_status.to_dict() or {}
        ).get("object_name")

    storage_client = storage.Client(
        project=project_id
    )
    bucket = storage_client.bucket(bucket_name)

    if not bucket.exists():
        raise RuntimeError(
            f"Cloud Storage bucket does not exist: "
            f"gs://{bucket_name}"
        )

    blob = bucket.blob(object_name)

    # Upload first. The published status is updated only after
    # the upload has completed successfully.
    blob.upload_from_filename(
        str(pdf_path),
        content_type="application/pdf",
    )

    blob.cache_control = (
        "no-store, max-age=0, no-transform"
    )
    blob.content_disposition = (
        'inline; filename="Zwischenrangliste.pdf"'
    )
    blob.patch()

    pdf_url = blob.public_url

    status_reference.set(
        {
            "published": True,
            "published_at": firestore.SERVER_TIMESTAMP,
            "pdf_url": pdf_url,
            "object_name": object_name,
            "source_filename": pdf_path.name,
            "updated_at": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )

    # Remove the previous PDF only after the new PDF and metadata
    # have been published successfully.
    if (
        previous_object_name
        and previous_object_name != object_name
    ):
        try:
            bucket.blob(previous_object_name).delete()
        except NotFound:
            pass
        except Exception as error:
            print(
                "Warning: the previous PDF could not be "
                f"deleted: {error}",
                file=sys.stderr,
            )

    print("Zwischenrangliste published successfully.")
    print(f"Published at: {now:%d.%m.%Y %H:%M Uhr}")
    print(f"PDF URL: {pdf_url}")


def unpublish(
    project_id: str,
    bucket_name: str | None,
) -> None:
    firestore_client = firestore.Client(
        project=project_id
    )
    status_reference = get_status_reference(
        firestore_client
    )

    current_status = status_reference.get()
    current_object_name = None

    if current_status.exists:
        current_object_name = (
            current_status.to_dict() or {}
        ).get("object_name")

    status_reference.set(
        {
            "published": False,
            "published_at": None,
            "pdf_url": None,
            "object_name": None,
            "source_filename": None,
            "updated_at": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )

    if current_object_name and bucket_name:
        storage_client = storage.Client(
            project=project_id
        )
        bucket = storage_client.bucket(bucket_name)

        try:
            bucket.blob(current_object_name).delete()
        except NotFound:
            pass
        except Exception as error:
            print(
                "Warning: the old PDF could not be "
                f"deleted: {error}",
                file=sys.stderr,
            )

    print("Live ranking status set to not published.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish or unpublish the current "
            "Zwischenrangliste."
        )
    )

    parser.add_argument(
        "--project",
        default=os.getenv("GOOGLE_CLOUD_PROJECT"),
        help=(
            "Google Cloud project ID. Defaults to "
            "GOOGLE_CLOUD_PROJECT."
        ),
    )

    parser.add_argument(
        "--bucket",
        default=os.getenv("LIVE_RANKING_BUCKET"),
        help=(
            "Cloud Storage bucket. Defaults to "
            "LIVE_RANKING_BUCKET."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    publish_parser = subparsers.add_parser(
        "publish",
        help="Publish a new Zwischenrangliste PDF.",
    )
    publish_parser.add_argument(
        "pdf",
        type=Path,
        help="Path to the PDF.",
    )

    subparsers.add_parser(
        "unpublish",
        help="Mark the Zwischenrangliste as unpublished.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if not arguments.project:
        raise RuntimeError(
            "Missing project ID. Set "
            "GOOGLE_CLOUD_PROJECT or use --project."
        )

    if arguments.command == "publish":
        if not arguments.bucket:
            raise RuntimeError(
                "Missing bucket name. Set "
                "LIVE_RANKING_BUCKET or use --bucket."
            )

        publish_pdf(
            pdf_path=arguments.pdf,
            project_id=arguments.project,
            bucket_name=arguments.bucket,
        )
        return

    unpublish(
        project_id=arguments.project,
        bucket_name=arguments.bucket,
    )


if __name__ == "__main__":
    main()