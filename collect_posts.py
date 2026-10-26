
import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import requests

DATASET_URL = (
    "https://huggingface.co/datasets/"
    "sentence-transformers/reddit-title-body/"
    "resolve/main/reddit_title_text_2021.jsonl.gz"
)

OUTPUT_FILE = Path("labels.csv")

TARGETS = {
    "question": 70,
    "opinion": 65,
    "information": 65,
}

QUESTION_WORDS = {
    "what", "why", "how", "when", "where", "which",
    "who", "does", "do", "did", "can", "could",
    "should", "would", "is", "are", "anyone",
}

OPINION_PHRASES = (
    "i think", "i feel", "i believe",
    "in my opinion", "personally",
    "i love", "i hate", "i prefer",
    "overrated", "underrated",
    "best game", "worst game",
    "disappointed", "amazing",
    "terrible", "favorite", "favourite",
    "better than", "worse than", "i wish",
)

INFORMATION_PHRASES = (
    "announced", "announcement", "released",
    "release date", "launches", "launching",
    "update", "patch notes", "trailer",
    "confirmed", "official", "available now",
    "coming to", "delayed", "developer",
    "publisher", "new dlc",
)


def classify_post(text):
    """Suggest a label using rules, not a language model."""
    lower = text.lower()
    first_word = re.split(r"\W+", lower)[0]

    scores = {
        "question": 0,
        "opinion": 0,
        "information": 0,
    }

    if "?" in text:
        scores["question"] += 3

    if first_word in QUESTION_WORDS:
        scores["question"] += 3

    for phrase in OPINION_PHRASES:
        if phrase in lower:
            scores["opinion"] += 2

    for phrase in INFORMATION_PHRASES:
        if phrase in lower:
            scores["information"] += 2

    if any(
        phrase in lower
        for phrase in (
            "recommend me",
            "recommendations?",
            "looking for a game",
            "any suggestions",
            "what game",
            "which game",
        )
    ):
        scores["question"] += 3

    label = max(scores, key=scores.get)

    if scores[label] == 0:
        return None

    # Exclude tied predictions rather than assign
    # arbitrary labels.
    if list(scores.values()).count(scores[label]) > 1:
        return None

    return label


def collect_posts():
    selected = {label: [] for label in TARGETS}
    seen = set()
    examined = 0

    print("Streaming historical Reddit submissions...")
    print("Filtering for r/gaming")
    print("Targets:", TARGETS)

    with requests.get(
        DATASET_URL,
        stream=True,
        timeout=(30, 120),
    ) as response:
        response.raise_for_status()

        response.raw.decode_content = False

        with gzip.GzipFile(fileobj=response.raw) as archive:
            for raw_line in archive:
                examined += 1

                try:
                    post = json.loads(raw_line)
                except (ValueError, UnicodeDecodeError):
                    continue

                subreddit = str(
                    post.get("subreddit", "")
                ).lower().removeprefix("r/")

                if subreddit != "gaming":
                    continue

                title = str(
                    post.get("title") or ""
                ).strip()

                body = str(
                    post.get("body") or ""
                ).strip()

                if not title:
                    continue

                if body.lower() in {
                    "[removed]", "[deleted]"
                }:
                    continue

                full_text = title

                if body:
                    full_text += "\n\n" + body

                normalized = re.sub(
                    r"\s+", " ", full_text
                ).strip().casefold()

                if normalized in seen:
                    continue

                label = classify_post(full_text)

                if label is None:
                    continue

                if len(selected[label]) >= TARGETS[label]:
                    continue

                seen.add(normalized)

                selected[label].append({
                    "text": full_text,
                    "label": label,
                    "note": (
                        "rule-assisted prelabel; "
                        "needs manual review"
                    ),
                })

                counts = {
                    name: len(rows)
                    for name, rows in selected.items()
                }

                total = sum(counts.values())

                if total % 20 == 0:
                    print(
                        f"Collected {total}/200:",
                        counts,
                        flush=True,
                    )

                if all(
                    len(selected[name]) == target
                    for name, target in TARGETS.items()
                ):
                    print(
                        "Target reached after examining",
                        examined,
                        "submissions."
                    )
                    return selected

    return selected


def main():
    try:
        selected = collect_posts()
    except (
        requests.RequestException,
        OSError,
        EOFError,
    ) as exc:
        print("Collection failed:", exc)
        print("Existing labels.csv was not changed.")
        return

    rows = []

    for label in TARGETS:
        rows.extend(selected[label])

    if len(rows) != sum(TARGETS.values()):
        print(
            "Only found", len(rows),
            "qualifying posts."
        )
        print(
            "Existing labels.csv was not changed."
        )
        return

    with OUTPUT_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["text", "label", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["label"] for row in rows)

    print("\nSaved:", OUTPUT_FILE.resolve())
    print("Total:", len(rows))
    print("Counts:", dict(counts))
    print(
        "All labels are preliminary and "
        "require manual review."
    )


if __name__ == "__main__":
    main()
