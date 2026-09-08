"""
eval_genre.py - Zero-shot Genre Classification Evaluation Harness

Evaluates zero-shot genre predictions against ground-truth human labels
without external dependencies (no sklearn, no pandas).

Usage:
  1. Generate evaluation sample:
     python eval_genre.py --sample 10

  2. Fill in the 'true_genre' column in eval_labels.csv

  3. Compute accuracy and confusion matrix:
     python eval_genre.py --evaluate
"""

import argparse
import csv
import os
import sqlite3
import sys

SQLITE_DB_PATH = "news_store.db"
EVAL_CSV_PATH = "eval_labels.csv"


def sample_segments(n: int, db_path: str = SQLITE_DB_PATH, out_path: str = EVAL_CSV_PATH) -> int:
    """Sample N random segments from SQLite and save to eval_labels.csv."""
    if not os.path.isfile(db_path):
        print(f"Error: Database file not found: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT faiss_id, text, genre FROM segments ORDER BY RANDOM() LIMIT ?", (n,)
    ).fetchall()
    conn.close()

    if not rows:
        print("Error: No segments found in database to sample.", file=sys.stderr)
        return 1

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["faiss_id", "text", "predicted_genre", "true_genre"])
        for r in rows:
            writer.writerow([r["faiss_id"], r["text"], r["genre"], ""])

    print(f"Sampled {len(rows)} segments to {out_path}.")
    print("Fill in the 'true_genre' column and run: python eval_genre.py --evaluate")
    return 0


def evaluate_labels(csv_path: str = EVAL_CSV_PATH) -> int:
    """Read labeled evaluation CSV and print accuracy and confusion matrix."""
    if not os.path.isfile(csv_path):
        print(f"Error: Evaluation file not found: {csv_path}", file=sys.stderr)
        print("Run 'python eval_genre.py --sample N' first.", file=sys.stderr)
        return 1

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    labeled_rows = [r for r in rows if r.get("true_genre", "").strip()]
    if not labeled_rows:
        print(f"No labeled rows found in {csv_path} (true_genre is empty).", file=sys.stderr)
        print("Please fill in ground truth values in the 'true_genre' column first.", file=sys.stderr)
        return 1

    # Unique genre labels
    all_genres = sorted(list(set(
        [r["true_genre"].strip() for r in labeled_rows] +
        [r["predicted_genre"].strip() for r in labeled_rows]
    )))

    # Confusion matrix: true_genre -> predicted_genre -> count
    matrix: dict[str, dict[str, int]] = {g: {p: 0 for p in all_genres} for g in all_genres}
    correct = 0

    for r in labeled_rows:
        t = r["true_genre"].strip()
        p = r["predicted_genre"].strip()
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
        if t.lower() == p.lower():
            correct += 1

    total = len(labeled_rows)
    accuracy = (correct / total) * 100.0 if total > 0 else 0.0

    print("=" * 70)
    print("  GENRE CLASSIFICATION EVALUATION REPORT")
    print("=" * 70)
    print(f"Total labeled samples: {total}")
    print(f"Correct predictions:   {correct}")
    print(f"Accuracy:              {accuracy:.2f}%\n")

    # Format confusion matrix
    col_width = max(max(len(g) for g in all_genres) + 2, 12)
    label_title = "True \\ Pred"
    header = f"{label_title:<{col_width}}" + "".join(f"{g[:col_width-1]:<{col_width}}" for g in all_genres)
    print("Confusion Matrix:")
    print(header)
    print("-" * len(header))
    for t in all_genres:
        row_str = f"{t:<{col_width}}"
        for p in all_genres:
            cnt = matrix[t][p]
            row_str += f"{cnt:<{col_width}}"
        print(row_str)

    print("=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Genre evaluation harness for AIR News Intelligence.")
    parser.add_argument("--sample", type=int, metavar="N", help="Sample N random segments into eval_labels.csv")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate accuracy from eval_labels.csv")
    args = parser.parse_args()

    if args.sample:
        return sample_segments(args.sample)
    elif args.evaluate:
        return evaluate_labels()
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
