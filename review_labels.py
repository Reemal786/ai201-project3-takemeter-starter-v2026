
import csv
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline

INPUT_FILE = Path("labels.csv")
OUTPUT_FILE = Path("label_review.csv")

MODEL_NAME = "facebook/bart-large-mnli"

LABEL_DESCRIPTIONS = {
    "question": (
        "a request for advice, recommendations, "
        "explanations, or information from other users"
    ),
    "opinion": (
        "a personal judgment, preference, criticism, "
        "or argument about gaming"
    ),
    "information": (
        "gaming news, updates, announcements, "
        "or factual information"
    ),
}


def main():
    if not INPUT_FILE.exists():
        print("ERROR: labels.csv does not exist.")
        return

    df = pd.read_csv(INPUT_FILE)

    required = {"text", "label", "note"}

    if not required.issubset(df.columns):
        print("ERROR: Missing required CSV columns.")
        return

    print(f"Loaded {len(df)} posts.")
    print("Original label distribution:")
    print(df["label"].value_counts())

    print("\nLoading pretrained classification model...")
    print("The first run may download model weights.")

    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=-1,
    )

    descriptions = list(LABEL_DESCRIPTIONS.values())
    description_to_label = {
        description: label
        for label, description in LABEL_DESCRIPTIONS.items()
    }

    results = []

    for index, row in df.iterrows():
        text = str(row["text"])

        # The original text is never modified.
        prediction = classifier(
            text,
            candidate_labels=descriptions,
            hypothesis_template="This post is primarily {}.",
            multi_label=False,
            truncation=True,
        )

        suggested_description = prediction["labels"][0]
        suggested_label = description_to_label[
            suggested_description
        ]

        confidence = float(prediction["scores"][0])

        original_label = str(row["label"])

        results.append({
            "text": text,
            "original_label": original_label,
            "suggested_label": suggested_label,
            "confidence": round(confidence, 4),
            "agreement": (
                "yes"
                if original_label == suggested_label
                else "no"
            ),
            "review_status": (
                "needs review"
                if confidence < 0.70
                or original_label != suggested_label
                else "preliminary agreement"
            ),
        })

        if (index + 1) % 10 == 0:
            print(
                f"Reviewed {index + 1}/{len(df)} posts.",
                flush=True,
            )

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "text",
                "original_label",
                "suggested_label",
                "confidence",
                "agreement",
                "review_status",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    counts = Counter(
        item["suggested_label"]
        for item in results
    )

    disagreements = sum(
        item["agreement"] == "no"
        for item in results
    )

    print("\nReview complete.")
    print("Saved:", OUTPUT_FILE.resolve())
    print("Suggested label counts:", dict(counts))
    print("Disagreements:", disagreements)
    print(
        "Original labels.csv has not been modified."
    )


if __name__ == "__main__":
    main()
