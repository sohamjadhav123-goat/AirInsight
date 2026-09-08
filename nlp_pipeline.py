"""
nlp_pipeline.py - NLP Pipeline for AIR Broadcast News Intelligence System (Phase 1)

Takes a transcript .txt file and produces classified, deduplicated, embedded
segments ready for FAISS indexing.

CLI usage:
  python nlp_pipeline.py --transcript output/text/2026-05-01/file.txt --date 2026-05-01 --hour 0800
  python nlp_pipeline.py --transcript ... --fast --mmr-threshold 0.90
"""

import argparse
import json
import logging
import os
import sys
from typing import Optional

import nltk
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("nlp_pipeline")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    logger.info("Downloading NLTK punkt_tab tokenizer data...")
    nltk.download("punkt_tab", quiet=True)

GENRE_LABELS = [
    "Politics", "Economy", "Science & Technology", "Sports",
    "International", "Defence & Security", "Health",
    "Environment", "Society & Culture", "Other",
]

DEFAULT_CLASSIFIER_MODEL = "facebook/bart-large-mnli"
FAST_CLASSIFIER_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_MMR_THRESHOLD = 0.85

_classifier_pipeline = None
_embedding_model = None


def segment_sentences(text: str) -> list[str]:
    """Split raw transcript text into individual sentences using NLTK."""
    text = text.lstrip('\ufeff')
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


def _get_classifier(fast: bool = False):
    """Lazily load and cache the zero-shot classification pipeline."""
    global _classifier_pipeline
    if _classifier_pipeline is not None:
        return _classifier_pipeline
    from transformers import pipeline as hf_pipeline
    model_name = FAST_CLASSIFIER_MODEL if fast else DEFAULT_CLASSIFIER_MODEL
    logger.info("Loading zero-shot classifier: %s", model_name)
    _classifier_pipeline = hf_pipeline("zero-shot-classification", model=model_name, device=-1)
    return _classifier_pipeline


def classify_genre(sentence: str, fast: bool = False) -> tuple[str, float]:
    """Classify a single sentence into one of the predefined genre labels."""
    classifier = _get_classifier(fast=fast)
    result = classifier(sentence, candidate_labels=GENRE_LABELS)
    return result["labels"][0], float(result["scores"][0])


def classify_genres_batch(sentences: list[str], fast: bool = False) -> list[tuple[str, float]]:
    """Batch-classify sentences for better throughput."""
    if not sentences:
        return []
    classifier = _get_classifier(fast=fast)
    results = classifier(sentences, candidate_labels=GENRE_LABELS, batch_size=8)
    if isinstance(results, dict):
        results = [results]
    return [(r["labels"][0], float(r["scores"][0])) for r in results]


def _get_embedding_model():
    """Lazily load and cache the sentence-transformers embedding model."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def embed_sentences(sentences: list[str]) -> np.ndarray:
    """Generate embeddings for a list of sentences. Returns shape (N, 384)."""
    model = _get_embedding_model()
    return model.encode(sentences, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)


def mmr_novelty_filter_detailed(
    sentences: list[str],
    embeddings: np.ndarray,
    existing_embeddings: Optional[np.ndarray] = None,
    existing_texts: Optional[list[str]] = None,
    threshold: float = DEFAULT_MMR_THRESHOLD,
) -> tuple[list[int], list[dict]]:
    """MMR novelty gate: streaming cosine-similarity deduplication.

    Returns:
        (novel_indices, rejections)
        where rejections is a list of dicts:
        {"rejected_text": str, "matched_text": str, "cosine_score": float, "matched_source_idx": int}
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms

    if existing_embeddings is not None and len(existing_embeddings) > 0:
        ex_norms = np.linalg.norm(existing_embeddings, axis=1, keepdims=True)
        ex_norms = np.where(ex_norms == 0, 1, ex_norms)
        pool = list(existing_embeddings / ex_norms)
        if existing_texts and len(existing_texts) == len(existing_embeddings):
            pool_texts = list(existing_texts)
        else:
            pool_texts = [f"Existing segment #{idx+1}" for idx in range(len(existing_embeddings))]
    else:
        pool = []
        pool_texts = []

    novel_indices: list[int] = []
    rejections: list[dict] = []

    for i in range(len(normed)):
        if pool:
            pool_matrix = np.vstack(pool)
            sims = pool_matrix @ normed[i]
            best_idx = int(np.argmax(sims))
            max_sim = float(sims[best_idx])
        else:
            max_sim = -1.0
            best_idx = -1

        if max_sim >= threshold:
            logger.debug("DUPLICATE (sim=%.3f): %s", max_sim, sentences[i][:80])
            matched_str = pool_texts[best_idx] if 0 <= best_idx < len(pool_texts) else ""
            rejections.append({
                "rejected_text": sentences[i],
                "matched_text": matched_str,
                "cosine_score": round(max_sim, 4),
                "matched_source_idx": best_idx,
            })
        else:
            novel_indices.append(i)
            pool.append(normed[i])
            pool_texts.append(sentences[i])

    return novel_indices, rejections


def mmr_novelty_filter(
    sentences: list[str],
    embeddings: np.ndarray,
    existing_embeddings: Optional[np.ndarray] = None,
    threshold: float = DEFAULT_MMR_THRESHOLD,
) -> list[int]:
    """MMR novelty gate: streaming cosine-similarity deduplication.

    Returns indices of novel (non-duplicate) sentences.
    """
    novel_indices, _ = mmr_novelty_filter_detailed(
        sentences, embeddings, existing_embeddings, threshold=threshold
    )
    return novel_indices


def process_transcript(
    text: str,
    existing_embeddings: Optional[np.ndarray] = None,
    fast: bool = False,
    mmr_threshold: float = DEFAULT_MMR_THRESHOLD,
    detailed: bool = False,
    existing_texts: Optional[list[str]] = None,
):
    """Run the full NLP pipeline on a transcript string.

    If detailed=True, returns (segments, rejections).
    Otherwise returns segments list.
    """
    sentences = segment_sentences(text)
    logger.info("Segmented transcript into %d sentences", len(sentences))
    if not sentences:
        return ([], []) if detailed else []

    embeddings = embed_sentences(sentences)
    logger.info("Generated embeddings: shape %s", embeddings.shape)

    novel_indices, rejections = mmr_novelty_filter_detailed(
        sentences,
        embeddings,
        existing_embeddings=existing_embeddings,
        existing_texts=existing_texts,
        threshold=mmr_threshold,
    )
    logger.info("MMR filter: %d novel / %d total (threshold=%.2f)", len(novel_indices), len(sentences), mmr_threshold)
    if not novel_indices:
        return ([], rejections) if detailed else []

    novel_sentences = [sentences[i] for i in novel_indices]
    novel_embeddings = embeddings[novel_indices]
    logger.info("Classifying %d novel sentences...", len(novel_sentences))
    genre_results = classify_genres_batch(novel_sentences, fast=fast)

    segments = []
    for idx, (genre, confidence) in enumerate(genre_results):
        segments.append({
            "text": novel_sentences[idx],
            "genre": genre,
            "genre_confidence": confidence,
            "embedding": novel_embeddings[idx],
        })
    return (segments, rejections) if detailed else segments


def main() -> int:
    """CLI entry point for standalone pipeline execution."""
    parser = argparse.ArgumentParser(description="NLP pipeline for AIR transcripts.")
    parser.add_argument("--transcript", required=True, help="Path to transcript .txt file")
    parser.add_argument("--date", required=True, help="Broadcast date YYYY-MM-DD")
    parser.add_argument("--hour", required=True, type=int, help="Broadcast hour e.g. 0800")
    parser.add_argument("--fast", action="store_true", help="Use lighter classifier model")
    parser.add_argument("--mmr-threshold", type=float, default=DEFAULT_MMR_THRESHOLD,
                        help=f"Cosine sim threshold for dedup (default {DEFAULT_MMR_THRESHOLD})")
    args = parser.parse_args()

    if not os.path.isfile(args.transcript):
        logger.error("Transcript file not found: %s", args.transcript)
        return 1

    with open(args.transcript, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        logger.warning("Transcript is empty: %s", args.transcript)
        return 0

    existing_embeddings = None
    try:
        from vector_store import get_all_embeddings
        existing_embeddings = get_all_embeddings()
        if existing_embeddings is not None and len(existing_embeddings) > 0:
            logger.info("Loaded %d existing embeddings for dedup", len(existing_embeddings))
    except Exception:
        logger.info("No existing FAISS index found - starting fresh")

    segments = process_transcript(text, existing_embeddings=existing_embeddings,
                                  fast=args.fast, mmr_threshold=args.mmr_threshold)
    logger.info("Pipeline complete: %d novel segments produced", len(segments))
    for seg in segments:
        print(json.dumps({
            "text": seg["text"], "genre": seg["genre"],
            "genre_confidence": round(seg["genre_confidence"], 4),
            "embedding_shape": list(seg["embedding"].shape),
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
