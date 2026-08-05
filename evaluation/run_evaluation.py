from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
TEST_FILE = BASE_DIR / "test_cases.csv"


def split_terms(value: str) -> list[str]:
    return [
        term.strip()
        for term in value.split("||")
        if term.strip()
    ]


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def evaluate_answer(
    answer: str,
    must_contain_all: list[str],
    must_contain_any: list[str],
    must_not_contain: list[str],
) -> tuple[bool, str]:
    normalized_answer = normalize(answer)
    failures = []

    missing_all = [
        term
        for term in must_contain_all
        if normalize(term) not in normalized_answer
    ]

    if missing_all:
        failures.append(
            f"Missing required terms: {missing_all}"
        )

    if must_contain_any:
        contains_any = any(
            normalize(term) in normalized_answer
            for term in must_contain_any
        )

        if not contains_any:
            failures.append(
                "None of the acceptable terms appeared: "
                f"{must_contain_any}"
            )

    forbidden_found = [
        term
        for term in must_not_contain
        if normalize(term) in normalized_answer
    ]

    if forbidden_found:
        failures.append(
            f"Forbidden terms found: {forbidden_found}"
        )

    return not failures, "; ".join(failures)


def send_question(
    service_url: str,
    question: str,
) -> tuple[str, float]:
    endpoint = f"{service_url.rstrip('/')}/chat"

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(
            {"message": question}
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    start_time = time.perf_counter()

    with urllib.request.urlopen(
        request,
        timeout=60,
    ) as response:
        response_data = json.loads(
            response.read().decode("utf-8")
        )

    latency_seconds = time.perf_counter() - start_time

    return (
        response_data.get("answer", ""),
        latency_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "service_url",
        help=(
            "Application URL, for example "
            "http://127.0.0.1:8080"
        ),
    )
    arguments = parser.parse_args()

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    output_file = (
        BASE_DIR / f"results-{timestamp}.csv"
    )

    with TEST_FILE.open(
        encoding="utf-8",
        newline="",
    ) as input_handle:
        test_cases = list(
            csv.DictReader(input_handle)
        )

    result_rows = []
    passed = 0

    for test_case in test_cases:
        test_id = test_case["id"]
        question = test_case["question"]

        print(f"\n[{test_id}] {question}")

        try:
            answer, latency = send_question(
                arguments.service_url,
                question,
            )

            automatic_pass, reason = evaluate_answer(
                answer=answer,
                must_contain_all=split_terms(
                    test_case["must_contain_all"]
                ),
                must_contain_any=split_terms(
                    test_case["must_contain_any"]
                ),
                must_not_contain=split_terms(
                    test_case["must_not_contain"]
                ),
            )

            status = (
                "PASS"
                if automatic_pass
                else "FAIL"
            )

            if automatic_pass:
                passed += 1

        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as error:
            answer = ""
            latency = 0.0
            status = "ERROR"
            reason = str(error)

        print(f"{status}: {answer}")
        print(f"Latency: {latency:.2f}s")

        result_rows.append(
            {
                **test_case,
                "status": status,
                "actual_answer": answer,
                "failure_reason": reason,
                "latency_seconds": (
                    f"{latency:.3f}"
                ),
                "manual_result": "",
                "manual_notes": "",
            }
        )

    fieldnames = list(result_rows[0].keys())

    with output_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_handle:
        writer = csv.DictWriter(
            output_handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(result_rows)

    total = len(test_cases)

    print("\n------------------------------")
    print(f"Automatic checks: {passed}/{total}")
    print(f"Results written to: {output_file}")


if __name__ == "__main__":
    main()