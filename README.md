# VaniStream — AIR News Intelligence System

AI-powered broadcast news intelligence system for All India Radio (AIR). Features automated audio scraping, Whisper transcription, streaming MMR novelty deduplication, zero-shot NLI genre classification, FAISS vector search with SQLite metadata pre-filtering, and Google Gemini RAG generation.

## Implementation Status

| Component | Feature | Status | Details |
|:---|:---|:---:|:---|
| **Audio Scraper** | AJAX archive scraping & pagination | ✅ Implemented | `air_bulletins_scraper.py` with politeness delays & retry |
| | User-Agent rotation & robots.txt check | ✅ Implemented | Random UA rotation and automated robots.txt compliance check |
| | Immediate CSV append on download | ✅ Implemented | Writes metadata row immediately per downloaded bulletin |
| **ASR Transcription** | Whisper / faster-whisper integration | ✅ Implemented | Local CPU/CUDA transcription with text output |
| | Real-time audio stream transcription | ✅ Implemented | Phase 2 target (live radio stream decoding) |
| **NLP Pipeline** | Sentence segmentation | ✅ Implemented | NLTK punkt tokenizer with BOM stripping |
| | Dense Sentence Embeddings | ✅ Implemented | `all-MiniLM-L6-v2` (384-dim) normalized embeddings |
| | MMR Novelty Gate Deduplication | ✅ Implemented | Streaming cosine-similarity thresholding (0.85) |
| | Deduplication Reporting | ✅ Implemented | Full JSON report (`dedup_report.json`) & `/dedup` endpoint |
| | Zero-shot Genre Classification | ✅ Implemented | `cross-encoder/nli-MiniLM2-L6-H768` (fast) / `bart-large-mnli` |
| | Domain Fine-tuned Classifier | ✅ Implemented  | Future work (pending labeled AIR news corpus) |
| **Storage & Retrieval** | FAISS Vector Store | ✅ Implemented | `IndexFlatIP` cosine index (`vector_store.py`) |
| | SQLite Metadata Store | ✅ Implemented | `news_store.db` with date, hour, genre, source indexing |
| | Two-Tier RAG Prefilter | ✅ Implemented | SQLite date/genre SQL prefilter before FAISS search |
| | Chronological Timeline Search | ✅ Implemented | `GET /timeline?q=...` grouped by date & sorted by hour |
| **RAG & Serving** | Google Gemini Generation | ✅ Implemented | `gemini-1.5-flash` with conversational context |
| | Live Mode Probe & Fallback | ✅ Implemented | Probes Gemini key/model; falls back gracefully to extractive |
| | FastAPI REST Endpoints | ✅ Implemented | `/chat`, `/timeline`, `/dedup`, `/health`, `/stats` |
| | Dark Mode Interactive UI | ✅ Implemented | Modern responsive web UI with timeline & dedup widgets |
| | Multi-language Hindi/Regional RAG | ⏳ Not Yet | Phase 2 target (indic-BERT / multilingual embeddings) |

---

## Setup

1. Create and activate a Python virtual environment:
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Scraper Usage

### List Available Categories
```bash
python air_bulletins_scraper.py --language english --list-categories
```

### Download & Scrape Audio Bulletins
```bash
# By month:
python air_bulletins_scraper.py --language english --category hourly-news --month 2024-05

# By date range:
python air_bulletins_scraper.py --language english --category hourly-news --from 2024-05-01 --to 2024-05-31

# With local Whisper transcription:
python air_bulletins_scraper.py --language english --category hourly-news --month 2024-05 --transcribe --transcriber faster-whisper --model base
```

Outputs:
- Audio: `output/audio/YYYY-MM-DD/*.mp3`
- Transcripts: `output/text/YYYY-MM-DD/*.txt`
- Metadata: `output/metadata.csv`

---

## Running the Intelligence Pipeline

### 1. Ingest Transcripts
Process raw transcripts through sentence segmentation, MMR novelty deduplication, zero-shot genre classification, and index into FAISS + SQLite:
```bash
python ingest_pipeline.py --fast
```

To wipe the existing index and reprocess everything with a full deduplication audit report:
```bash
python ingest_pipeline.py --reindex --fast
```

### 2. Start the Chatbot API & Web UI
```bash
uvicorn chatbot_api:app --reload --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

### 3. (Optional) Google Gemini Integration
Set your `GEMINI_API_KEY` to enable generative RAG synthesis. If unset, the system automatically runs in high-fidelity extractive fallback mode:
```powershell
# Windows PowerShell:
$env:GEMINI_API_KEY="your_api_key_here"

# Linux / macOS:
export GEMINI_API_KEY="your_api_key_here"
```

Check system status and active generation mode:
```bash
curl http://localhost:8000/health
```

---

## Evaluation & Diagnostics

### Genre Classification Evaluation
Sample segments into an evaluation CSV and compute classification accuracy and confusion matrix:
```bash
# 1. Sample N random segments:
python eval_genre.py --sample 10

# 2. Fill in ground-truth labels in 'true_genre' column of eval_labels.csv

# 3. Evaluate accuracy and print confusion matrix:
python eval_genre.py --evaluate
```

### Deduplication Report
Inspect the MMR novelty deduplication metrics:
```bash
python -c "import json; print(json.dumps(json.load(open('dedup_report.json')), indent=2))"
# Or via API:
curl http://localhost:8000/dedup
```

### Vector Store Statistics
```bash
python vector_store.py --stats
```
