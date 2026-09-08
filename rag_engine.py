"""
rag_engine.py - Two-tier RAG Engine for AIR News Intelligence System

Implements:
  1. Query parsing (date range + genre extraction)
  2. Metadata pre-filtering via SQLite
  3. Semantic search via FAISS
  4. LLM answer generation via Google Gemini (with extractive fallback)

Usage (as module):
  from rag_engine import answer_query
  result = answer_query("What happened in Indian politics last week?")
"""

import datetime
import logging
import os
import re
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("rag_engine")

# Genre keyword mapping for Phase 1 (word-boundary matched)
GENRE_KEYWORD_MAP = {
    "cricket": "Sports", "football": "Sports", "hockey": "Sports",
    "tennis": "Sports", "olympics": "Sports", "sports": "Sports",
    "match": "Sports", "tournament": "Sports", "athlete": "Sports",
    "parliament": "Politics", "election": "Politics", "minister": "Politics",
    "modi": "Politics", "bjp": "Politics", "congress": "Politics",
    "political": "Politics", "politics": "Politics", "vote": "Politics",
    "lok sabha": "Politics", "rajya sabha": "Politics", "bill": "Politics",
    "economy": "Economy", "gdp": "Economy", "inflation": "Economy",
    "rbi": "Economy", "rupee": "Economy", "stock": "Economy",
    "market": "Economy", "budget": "Economy", "trade": "Economy",
    "fiscal": "Economy", "tax": "Economy", "economic": "Economy",
    "science": "Science & Technology", "technology": "Science & Technology",
    "isro": "Science & Technology", "space": "Science & Technology",
    "artificial intelligence": "Science & Technology", "digital": "Science & Technology",
    "research": "Science & Technology", "satellite": "Science & Technology",
    "china": "International", "pakistan": "International",
    "america": "International", "russia": "International",
    "united nations": "International", "united states": "International",
    "international": "International", "global": "International",
    "foreign": "International", "diplomat": "International",
    "army": "Defence & Security", "navy": "Defence & Security",
    "military": "Defence & Security", "defence": "Defence & Security",
    "defense": "Defence & Security", "security": "Defence & Security",
    "terror": "Defence & Security", "border": "Defence & Security",
    "health": "Health", "covid": "Health", "hospital": "Health",
    "disease": "Health", "vaccine": "Health", "medical": "Health",
    "doctor": "Health", "pandemic": "Health",
    "environment": "Environment", "climate": "Environment",
    "pollution": "Environment", "forest": "Environment",
    "wildlife": "Environment", "green": "Environment", "flood": "Environment",
    "drought": "Environment", "cyclone": "Environment", "weather": "Environment",
    "culture": "Society & Culture", "festival": "Society & Culture",
    "education": "Society & Culture", "society": "Society & Culture",
    "women": "Society & Culture", "caste": "Society & Culture",
    "religion": "Society & Culture",
}

_cached_anchor: Optional[datetime.date] = None


def _get_corpus_anchor() -> datetime.date:
    """Return the date anchor for relative query parsing.

    Uses MAX(date) from the vector store if available, falling back to
    datetime.date.today() if the database is empty.
    """
    global _cached_anchor
    if _cached_anchor is not None:
        return _cached_anchor

    import vector_store
    _, max_d = vector_store.get_date_range()
    if max_d:
        try:
            _cached_anchor = datetime.date.fromisoformat(max_d)
            logger.info("Date anchor: %s (corpus max)", _cached_anchor)
            return _cached_anchor
        except ValueError:
            pass

    _cached_anchor = datetime.date.today()
    logger.info("Date anchor: %s (real today, empty DB)", _cached_anchor)
    return _cached_anchor


def _parse_date_range(query: str) -> tuple[Optional[str], Optional[str]]:
    """Extract date range from natural language query.

    Handles: 'last N days', 'last week', 'yesterday', 'today',
    'on DD Month YYYY', 'between X and Y'.
    """
    today = _get_corpus_anchor()
    q = query.lower()

    # "last N days"
    m = re.search(r"last\s+(\d+)\s+days?", q)
    if m:
        n = int(m.group(1))
        return (today - datetime.timedelta(days=n)).isoformat(), today.isoformat()

    # "last week"
    if "last week" in q:
        return (today - datetime.timedelta(days=7)).isoformat(), today.isoformat()

    # "last month"
    if "last month" in q:
        return (today - datetime.timedelta(days=30)).isoformat(), today.isoformat()

    # "yesterday"
    if "yesterday" in q:
        yest = today - datetime.timedelta(days=1)
        return yest.isoformat(), yest.isoformat()

    # "today"
    if "today" in q:
        return today.isoformat(), today.isoformat()

    # "on 5th May" / "on May 5" / "on 2026-05-05"
    m = re.search(r"on\s+(\d{4}-\d{2}-\d{2})", q)
    if m:
        return m.group(1), m.group(1)

    m = re.search(r"on\s+(\d{1,2})(?:st|nd|rd|th)?\s+(\w+)(?:\s+(\d{4}))?", q)
    if m:
        day = int(m.group(1))
        month_str = m.group(2)
        year = int(m.group(3)) if m.group(3) else today.year
        try:
            month = datetime.datetime.strptime(month_str, "%B").month
        except ValueError:
            try:
                month = datetime.datetime.strptime(month_str, "%b").month
            except ValueError:
                return None, None
        d = datetime.date(year, month, day).isoformat()
        return d, d

    # "between X and Y"
    m = re.search(r"between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})", q)
    if m:
        return m.group(1), m.group(2)

    return None, None


def _extract_genre(query: str) -> Optional[str]:
    """Extract genre from query using word-boundary keyword matching.

    Returns None if no keywords match or if there is a tie between genres.
    """
    q = query.lower()
    genre_scores: dict[str, int] = {}
    for keyword, genre in GENRE_KEYWORD_MAP.items():
        # Use word boundaries to avoid substring false positives
        pattern = r'\b' + re.escape(keyword) + r'\b'
        hits = len(re.findall(pattern, q))
        if hits > 0:
            genre_scores[genre] = genre_scores.get(genre, 0) + hits
    if not genre_scores:
        return None
    # Return None on a tie instead of guessing
    top_score = max(genre_scores.values())
    top_genres = [g for g, s in genre_scores.items() if s == top_score]
    if len(top_genres) > 1:
        return None
    return top_genres[0]


def _build_context(sources: list[dict]) -> str:
    """Build the context string from retrieved segments."""
    lines = []
    for i, src in enumerate(sources, 1):
        lines.append(f"[{i}] ({src['date']}, {src['genre']}) {src['text']}")
    return "\n".join(lines)


def _generate_with_gemini(context: str, user_query: str, history: list[dict] = None) -> str:
    """Call Gemini API with system prompt, context, and user query."""
    try:
        import google.generativeai as genai
    except ImportError:
        logger.warning("google-generativeai not installed, using extractive fallback")
        return ""

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.info("GEMINI_API_KEY not set, using extractive fallback")
        return ""

    genai.configure(api_key=api_key)

    system_prompt = (
        "You are VaniStream, an AI news assistant for Indian broadcast news. "
        "Answer the user's question using ONLY the provided news context. "
        "Be concise, factual, and cite the date of the news. "
        "If the context doesn't contain the answer, say "
        "\"I don't have information about that in my news database.\""
    )

    # Build conversation messages
    contents = []
    if history:
        for turn in history[-3:]:  # last 3 turns for context continuity
            contents.append({"role": turn["role"], "parts": [turn["content"]]})

    user_message = f"Context:\n{context}\n\nUser Question: {user_query}"
    contents.append({"role": "user", "parts": [user_message]})

    try:
        model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_prompt)
        response = model.generate_content(contents)
        return response.text
    except Exception as e:
        logger.error("Gemini API error: %s", e)
        return ""


def answer_query(
    user_query: str,
    date_from: str = None,
    date_to: str = None,
    history: list[dict] = None,
) -> dict:
    """Two-tier RAG: metadata prefilter -> semantic search -> LLM generation.

    Args:
        user_query: Natural language question.
        date_from: Override start date (YYYY-MM-DD). Auto-parsed if None.
        date_to: Override end date (YYYY-MM-DD). Auto-parsed if None.
        history: Conversation history for context continuity.

    Returns:
        Dict with keys: answer, sources, filters_applied.
    """
    import vector_store
    from nlp_pipeline import embed_sentences

    # Step 1: Parse query for date range and genre
    if date_from is None or date_to is None:
        parsed_from, parsed_to = _parse_date_range(user_query)
        date_from = date_from or parsed_from
        date_to = date_to or parsed_to

    # Fallback: last 30 days if no date range found
    if not date_from or not date_to:
        today = _get_corpus_anchor()
        date_from = (today - datetime.timedelta(days=30)).isoformat()
        date_to = today.isoformat()

    genre = _extract_genre(user_query)

    logger.info("Query filters: date_from=%s, date_to=%s, genre=%s", date_from, date_to, genre)

    # Step 2: Metadata prefilter
    faiss_ids = vector_store.metadata_prefilter(date_from, date_to, genre)
    logger.info("Metadata prefilter returned %d candidate IDs", len(faiss_ids))

    filters_applied = {"date_from": date_from, "date_to": date_to, "genre": genre}

    if not faiss_ids:
        corpus_min, corpus_max = vector_store.get_date_range()
        genre_clause = f" in genre '{genre}'" if genre else ""
        if corpus_min and corpus_max:
            corpus_msg = f" My corpus covers {corpus_min} to {corpus_max}."
        else:
            corpus_msg = " My corpus is currently empty."
        return {
            "answer": f"No news segments found for {date_from} to {date_to}{genre_clause}.{corpus_msg}",
            "sources": [],
            "filters_applied": filters_applied,
        }

    # Step 3: Semantic search
    query_embedding = embed_sentences([user_query])[0]
    sources = vector_store.semantic_search(query_embedding, faiss_ids, top_k=5)
    logger.info("Semantic search returned %d results", len(sources))

    # Step 4: Generate answer
    if not sources:
        return {
            "answer": "I don't have any relevant news segments in my database for that query.",
            "sources": [],
            "filters_applied": filters_applied,
        }

    context = _build_context(sources)

    # Try LLM generation
    llm_answer = _generate_with_gemini(context, user_query, history)

    if llm_answer:
        answer = llm_answer
    else:
        # Extractive fallback
        answer = "Here are the most relevant news segments:\n\n" + context

    return {
        "answer": answer,
        "sources": sources,
        "filters_applied": filters_applied,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your question: ")
    result = answer_query(query)
    print("\n" + "=" * 60)
    print("ANSWER:", result["answer"])
    print("\nFILTERS:", result["filters_applied"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  [{s['date']}] ({s['genre']}, score={s['score']:.3f}) {s['text'][:100]}...")
