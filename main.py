import logging
import os

from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from google.cloud import firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
from pydantic import BaseModel, Field
from google.cloud.firestore_v1.base_query import FieldFilter

from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


app = FastAPI(
    title="Chneble Assistant",
    version="0.3.0",
)

GENERATION_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite",
)

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768
COLLECTION_NAME = "knowledge_chunks"
RETRIEVAL_LIMIT = 4

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_FILE = BASE_DIR / "static" / "index.html"

genai_client = genai.Client(
    http_options=types.HttpOptions(api_version="v1")
)

firestore_client = firestore.Client()
collection = firestore_client.collection(COLLECTION_NAME)


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=500,
    )

RESULTS_COLLECTION = "results"

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

def search_results(
    year: int,
    rank: int | None = None,
) -> list[dict]:
    """Retrieve ranking records using exact Firestore filters."""

    query = firestore_client.collection(
        RESULTS_COLLECTION
    ).where(
        filter=FieldFilter("year", "==", year)
    )

    if rank is not None:
        query = query.where(
            filter=FieldFilter("rank", "==", rank)
        )

    results = []

    for document in query.limit(100).stream():
        data = document.to_dict()

        results.append(
            {
                "year": data["year"],
                "category": data["category"],
                "rank": data.get("rank"),
                "rank_label": data.get("rank_label"),
                "row_text": data["row_text"],
                "source_pdf": data["source_pdf"],
            }
        )

    return sorted(
        results,
        key=lambda result: (
            result["category"],
            result["rank"]
            if result["rank"] is not None
            else 9999,
        ),
    )


def create_query_embedding(text: str) -> list[float]:
    """Create an embedding optimized for retrieval queries."""
    response = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIMENSION,
        ),
    )

    if not response.embeddings:
        raise RuntimeError("Gemini returned no query embedding.")

    return response.embeddings[0].values


def retrieve_chunks(
    question: str,
    limit: int = RETRIEVAL_LIMIT,
) -> list[dict]:
    """Retrieve the Firestore chunks nearest to the question."""
    query_embedding = create_query_embedding(question)

    vector_query = collection.find_nearest(
        vector_field="embedding",
        query_vector=Vector(query_embedding),
        distance_measure=DistanceMeasure.COSINE,
        limit=limit,
        distance_result_field="distance",
    )

    results: list[dict] = []

    for document in vector_query.stream():
        data = document.to_dict()

        results.append(
            {
                "text": data["text"],
                "source": data["source"],
                "filename": data["filename"],
                "page": data.get("page"),
                "distance": data.get("distance"),
            }
        )

    return results


RESULTS_COLLECTION = "results"


def search_knowledge(query: str) -> list[dict]:
    """Search the official Chneble rules and general information.

    Use this tool for questions about:
    - rules
    - date, time and location
    - registration
    - prices
    - sponsors
    - payment and food

    Args:
        query: The user's natural-language question.

    Returns:
        Relevant passages with their source information.
    """
    chunks = retrieve_chunks(query)

    return [
        {
            "text": chunk["text"],
            "source": chunk["source"],
            "filename": chunk.get("filename"),
            "page": chunk.get("page"),
        }
        for chunk in chunks
    ]


def search_results(
    year: int,
    rank: int | None = None,
) -> list[dict]:
    """Search historical Chneble rankings.

    Use this tool for questions about rankings, winners,
    placements, teams or participants from a specific year.

    Args:
        year: The competition year, for example 2017.
        rank: Optional numeric rank, for example 1 for winners.
            Omit it to retrieve all ranking entries for the year.

    Returns:
        Ranking records from the requested year.
    """
    query = firestore_client.collection(
        RESULTS_COLLECTION
    ).where(
        filter=FieldFilter("year", "==", year)
    )

    if rank is not None:
        query = query.where(
            filter=FieldFilter("rank", "==", rank)
        )

    results = []

    for document in query.limit(200).stream():
        data = document.to_dict()

        results.append(
            {
                "year": data["year"],
                "category": data["category"],
                "rank": data.get("rank"),
                "rank_label": data.get("rank_label"),
                "row_text": data["row_text"],
                "source_pdf": data["source_pdf"],
            }
        )

    return results


def build_context(chunks: list[dict]) -> str:
    sections: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        page_label = (
            f", page {chunk['page']}"
            if chunk["page"] is not None
            else ""
        )

        sections.append(
            f"[Source {index}: {chunk['source']}{page_label}]\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(sections)

@app.get("/results/{year}")
def get_results(
    year: int,
    rank: int | None = None,
) -> dict:
    results = search_results(
        year=year,
        rank=rank,
    )

    return {
        "year": year,
        "rank": rank,
        "count": len(results),
        "results": results,
    }


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_FILE)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": app.version,
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    question = request.message.strip()

    try:
        response = genai_client.models.generate_content(
            model=GENERATION_MODEL,
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are the official information assistant for the "
                    "Chneblermeisterschaft Schwyz. "

                    "For questions about rules, prices, sponsors, event "
                    "information, registration, food or payment, use the "
                    "search_knowledge tool. "

                    "For questions about historical rankings, winners, "
                    "placements, teams or participants, use the "
                    "search_results tool. "

                    "Use only information returned by the tools. "
                    "Never invent missing information. "

                    "Ranking row_text values may combine team names, "
                    "participant names and scores. Reproduce these values "
                    "conservatively and do not guess how they are separated. "

                    "If several categories match, clearly distinguish them. "
                    "Mention the relevant source file or page. "
                    "Answer in the same language as the user. "
                    "Keep the answer concise."

                    "When answering in German, use Swiss Standard German. "
                    "Write 'ss' instead of 'ß' and prefer terminology commonly used in Switzerland. "
                ),
                tools=[
                    search_knowledge,
                    search_results,
                ],
                automatic_function_calling=(
                    types.AutomaticFunctionCallingConfig(
                        maximum_remote_calls=4,
                    )
                ),
                temperature=0.0,
                max_output_tokens=400,
            ),
        )

        logging.info(
            "Automatic tool history: %s",
            response.automatic_function_calling_history,
        )

    except Exception:
        logging.exception("Gemini tool request failed")

        raise HTTPException(
            status_code=502,
            detail="The Gemini request failed.",
        )

    return {
        "answer": response.text or "No answer was generated.",
        "model": GENERATION_MODEL,
    }
