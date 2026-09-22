"""
ingest_pipeline.py - Orchestrator for AIR News Intelligence ingestion

Reads metadata.csv, finds unprocessed transcripts, runs the NLP pipeline,
and stores results in the FAISS + SQLite vector store.

CLI usage:
  python ingest_pipeline.py               # process all unprocessed files
  python ingest_pipeline.py --reindex     # wipe FAISS + DB and reindex everything
"""

import argparse
import csv
import json
import logging
import os
import re
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("ingest_pipeline")

METADATA_CSV = os.path.join("output", "metadata.csv")
PROCESSED_FILE = "processed_files.txt"


def _load_processed(path: str = PROCESSED_FILE) -> set[str]:
    """Load set of already-processed transcript file paths."""
    if not os.path.isfile(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _mark_processed(filepath: str, path: str = PROCESSED_FILE) -> None:
    """Append a transcript path to the processed-files tracker."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(filepath + "\n")


def _extract_hour(filename: str) -> int:
    """Extract 4-digit broadcast hour from filename, e.g. 'English-0800-0810-1.txt' -> 800."""
    m = re.search(r"(?<!\d)(\d{4})(?!\d)", filename)
    if m:
        return int(m.group(1))
    return 0


def _read_metadata(csv_path: str = METADATA_CSV) -> list[dict]:
    """Read metadata.csv and return rows that have a transcript_file."""
    if not os.path.isfile(csv_path):
        logger.warning("Metadata CSV not found: %s", csv_path)
        return []
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tf = (row.get("transcript_file") or "").strip()
            if tf:
                rows.append(row)
    return rows


def run(
    reindex: bool = False,
    fast: bool = False,
    mmr_threshold: float = 0.85,
    metadata_csv: str = METADATA_CSV,
    classifier: str = "zero-shot",
) -> dict:
    """Run the full ingestion pipeline.

    Args:
        reindex: If True, wipe the vector store and reprocess everything.
        fast: Use the lightweight NLP classifier (shorthand for classifier='fast').
        mmr_threshold: MMR cosine similarity threshold for deduplication.
        metadata_csv: Path to the metadata CSV.
        classifier: Classifier backend -- 'zero-shot' | 'fast' | 'finetuned'.
                    --fast flag maps to 'fast' for backward compatibility.

    Returns:
        Summary dict: files_processed, segments_added, duplicates_filtered.
    """
    import vector_store
    from nlp_pipeline import process_transcript

    # Backward compat: --fast maps to classifier='fast'
    if fast and classifier == "zero-shot":
        classifier = "fast"
    logger.info("Classifier backend: %s", classifier)

    if reindex:
        logger.info("REINDEX mode: wiping existing FAISS index and SQLite DB")
        vector_store.wipe()
        # Clear processed tracker
        if os.path.isfile(PROCESSED_FILE):
            os.remove(PROCESSED_FILE)
        if os.path.isfile("dedup_report.json"):
            os.remove("dedup_report.json")

    rows = _read_metadata(metadata_csv)
    if not rows:
        logger.info("No transcript rows found in metadata CSV")
        return {"files_processed": 0, "segments_added": 0, "duplicates_filtered": 0}

    processed = set() if reindex else _load_processed()
    logger.info("Found %d transcript rows, %d already processed", len(rows), len(processed))

    total_segments = 0
    total_duplicates = 0
    files_done = 0
    per_file: list[dict] = []
    all_rejections: list[dict] = []

    for row in rows:
        transcript_rel = row["transcript_file"]
        transcript_path = os.path.join("output", transcript_rel)

        if transcript_rel in processed:
            continue

        if not os.path.isfile(transcript_path):
            logger.warning("Transcript file missing: %s", transcript_path)
            continue

        date = row.get("date", "unknown")
        hour = _extract_hour(os.path.basename(transcript_path))

        with open(transcript_path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            logger.warning("Empty transcript: %s", transcript_path)
            _mark_processed(transcript_rel)
            continue

        # Load current embeddings and texts for dedup
        existing_embeddings = vector_store.get_all_embeddings()
        existing_texts = vector_store.get_all_texts()

        # Run NLP pipeline with detailed dedup reporting
        logger.info("Processing: %s (date=%s, hour=%d)", transcript_rel, date, hour)
        segments, file_rejections = process_transcript(
            text,
            existing_embeddings=existing_embeddings,
            existing_texts=existing_texts,
            fast=fast,
            mmr_threshold=mmr_threshold,
            detailed=True,
            classifier=classifier,
        )

        from nlp_pipeline import segment_sentences
        original_count = len(segment_sentences(text))
        novel_count = len(segments)
        rejected_count = len(file_rejections)
        total_duplicates += rejected_count

        per_file.append({
            "source_file": transcript_rel,
            "total_sentences": original_count,
            "novel_sentences": novel_count,
            "rejected_sentences": rejected_count,
        })
        all_rejections.extend(file_rejections)

        # Store in vector store
        if segments:
            added = vector_store.add_segments(segments, date, hour, transcript_rel)
            total_segments += added

        _mark_processed(transcript_rel)
        files_done += 1

    if per_file:
        tot_sentences = sum(f["total_sentences"] for f in per_file)
        nov_sentences = sum(f["novel_sentences"] for f in per_file)
        rej_sentences = sum(f["rejected_sentences"] for f in per_file)
        red_pct = round((rej_sentences / tot_sentences * 100), 1) if tot_sentences > 0 else 0.0

        all_rejections.sort(key=lambda x: x["cosine_score"], reverse=True)
        top_rejected = all_rejections[:20]

        dedup_report = {
            "totals": {
                "total_sentences": tot_sentences,
                "novel_sentences": nov_sentences,
                "rejected_sentences": rej_sentences,
                "reduction_pct": red_pct,
            },
            "per_file": per_file,
            "top_rejected_pairs": top_rejected,
        }
        with open("dedup_report.json", "w", encoding="utf-8") as f:
            json.dump(dedup_report, f, indent=2, ensure_ascii=False)
        logger.info("Saved dedup report to dedup_report.json (%d rejected pairs)", len(all_rejections))

    summary = {
        "files_processed": files_done,
        "segments_added": total_segments,
        "duplicates_filtered": total_duplicates,
    }
    logger.info(
        "Ingestion complete: %d files processed, %d segments added, %d duplicates filtered",
        summary["files_processed"],
        summary["segments_added"],
        summary["duplicates_filtered"],
    )
    return summary


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Ingest transcripts into the vector store.")
    parser.add_argument("--reindex", action="store_true",
                        help="Wipe FAISS + DB and reindex everything")
    parser.add_argument("--fast", action="store_true",
                        help="Use lighter NLP classifier model (shorthand for --classifier fast)")
    parser.add_argument(
        "--classifier",
        choices=["zero-shot", "fast", "finetuned"],
        default="zero-shot",
        help="Classifier backend: zero-shot (default), fast, or finetuned",
    )
    parser.add_argument("--mmr-threshold", type=float, default=0.85,
                        help="MMR cosine similarity threshold (default 0.85)")
    parser.add_argument("--csv", default=METADATA_CSV,
                        help="Path to metadata CSV")
    args = parser.parse_args()

    summary = run(
        reindex=args.reindex,
        fast=args.fast,
        mmr_threshold=args.mmr_threshold,
        metadata_csv=args.csv,
        classifier=args.classifier,
    )
    print(f"\n{'='*50}")
    print(f"  Files processed:     {summary['files_processed']}")
    print(f"  Segments added:      {summary['segments_added']}")
    print(f"  Duplicates filtered: {summary['duplicates_filtered']}")
    print(f"{'='*50}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
