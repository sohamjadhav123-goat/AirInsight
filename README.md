# VaniStream — AIR News Intelligence System

AI-powered broadcast news intelligence system for All India Radio (AIR). Features automated audio scraping, Whisper transcription, streaming MMR novelty deduplication, zero-shot NLI genre classification, FAISS vector search with SQLite metadata pre-filtering, and Google Gemini RAG generation.

## Implementation Status

| Component | Feature | Status | Details |
|:---|:---|:---:|:---|
| **Audio Scraper** | AJAX archive scraping & pagination | ✅ Implemented | `air_bulletins_scraper.py` with politeness delays & retry |
| | User-Agent rotation & robots.txt check | ✅ Implemented | Random UA rotation and automated robots.txt compliance check |
| | Immediate CSV append on download | ✅ Implemented | Writes metadata row immediately per downloaded bulletin |
| **ASR Transcription** | Whisper / faster-whisper integration | ✅ Implemented | Local CPU/CUDA transcription with text output |
| | Real-time audio stream transcription | ✅ Implemented | Live radio stream ffmpeg decoding, VAD transcription, MMR dedup & instant vector ingestion (`live_stream_ingest.py`) |
| **NLP Pipeline** | Sentence segmentation | ✅ Implemented | NLTK punkt tokenizer with BOM stripping |
| | Dense Sentence Embeddings | ✅ Implemented | `all-MiniLM-L6-v2` (384-dim) normalized embeddings |
| | MMR Novelty Gate Deduplication | ✅ Implemented | Streaming cosine-similarity thresholding (0.85) |
| | Deduplication Reporting | ✅ Implemented | Full JSON report (`dedup_report.json`) & `/dedup` endpoint |
| | Zero-shot Genre Classification | ✅ Implemented | `cross-encoder/nli-MiniLM2-L6-H768` (fast) / `bart-large-mnli` |
| | Domain Fine-tuned Classifier | ✅ Implemented | DistilBERT fine-tuned on AIR corpus; graceful fallback to zero-shot when checkpoint absent (`build_genre_training_data.py` → `train_genre_classifier.py`) |
| **Storage & Retrieval** | FAISS Vector Store | ✅ Implemented | `IndexFlatIP` cosine index (`vector_store.py`) |
| | SQLite Metadata Store | ✅ Implemented | `news_store.db` with date, hour, genre, source indexing |
| | Two-Tier RAG Prefilter | ✅ Implemented | SQLite date/genre SQL prefilter before FAISS search |
| | Chronological Timeline Search | ✅ Implemented | `GET /timeline?q=...` grouped by date & sorted by hour |
| **RAG & Serving** | Google Gemini Generation | ✅ Implemented | `gemini-1.5-flash` with conversational context |
| | Live Mode Probe & Fallback | ✅ Implemented | Probes Gemini key/model; falls back gracefully to extractive |
| | FastAPI REST Endpoints | ✅ Implemented | `/chat`, `/timeline`, `/dedup`, `/health`, `/stats`, `/stream/*` |
| | Dark Mode Interactive UI | ✅ Implemented | Modern responsive web UI with LIVE pill, timeline & dedup widgets |
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

3. Host Prerequisite (FFmpeg):
`ffmpeg` must be installed on your host system and accessible in your `PATH`:
- **Windows**: `winget install Gyan.FFmpeg` or `choco install ffmpeg`
- **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`
- Verify installation: `ffmpeg -version`

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

---

## Real-Time Audio Stream Transcription

In addition to batch archive scraping, VaniStream supports real-time transcription and continuous ingestion of live radio streams.

### Locating AIR Live Stream URLs
The ingestion engine is URL-agnostic and supports any Icecast HTTP MP3 stream or HLS `.m3u8` feed. To find AIR's live broadcast endpoint:
1. Navigate to [newsonair.gov.in](https://newsonair.gov.in) and open the **Live Radio** section.
2. Select your desired station (e.g. *AIR Gold*, *AIR Vividh Bharati*, or *AIR National News*).
3. Open Developer Tools in your browser (`F12` or `Ctrl+Shift+I`) and switch to the **Network** tab.
4. Filter by `media`, `.m3u8`, or `.mp3` and start playback.
5. Copy the active streaming URL and pass it to `--url` or `/stream/start`.

### Standalone CLI Execution
Run the live transcription loop directly from the terminal:
```bash
# Basic run with 30s rolling window, 5s overlap, faster-whisper VAD, and fast genre classification:
python live_stream_ingest.py --url "http://stream-url.example/live.mp3" --chunk-seconds 30 --overlap-seconds 5 --fast

# With custom model and language:
python live_stream_ingest.py --url "https://stream-url.example/live.m3u8" --language english --model base --device cpu
```

Output conventions match the batch pipeline:
- Rolling audio snapshots: `output/audio/<YYYY-MM-DD>/Live-<Language>-<HHMM>-<ID>.mp3`
- Transcripts: `output/text/<YYYY-MM-DD>/Live-<Language>-<HHMM>-<ID>.txt`
- Appends metadata row to `output/metadata.csv`
- Immediately indexes novel segments into FAISS & SQLite (searchable in RAG within seconds).

---

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

# 4. Compare zero-shot vs fine-tuned (side-by-side matrices + accuracy delta):
python eval_genre.py --compare
```

### Domain Fine-tuned Genre Classifier
The fine-tuned classifier is a selectable alternative to zero-shot NLI. Zero-shot remains the default.

**Step 1 — Bootstrap training data:**
```bash
# Silver-label existing DB segments + transcript files + synthetic augmentation:
python build_genre_training_data.py --augment
```
This writes `genre_training_data.csv` with columns: `text, predicted_genre, confidence, true_genre`.
> ⚠️ **Human review required**: Open `genre_training_data.csv` and fill in the `true_genre` column before training. The `predicted_genre` values are zero-shot silver labels and may be incorrect.

**Step 2 — Train the classifier:**
```bash
python train_genre_classifier.py
# Optional flags:
python train_genre_classifier.py --epochs 10 --batch-size 32 --model distilbert-base-uncased
```
Saves the best checkpoint (by val accuracy) to `models/air-genre-classifier/` and prints per-class F1.

**Step 3 — Use the fine-tuned classifier:**
```bash
# Ingest pipeline:
python ingest_pipeline.py --classifier finetuned

# Standalone NLP pipeline:
python nlp_pipeline.py --transcript output/text/2026-05-01/file.txt --date 2026-05-01 --hour 8 --classifier finetuned

# Compare vs zero-shot:
python eval_genre.py --compare
```
> **Graceful degradation**: If `models/air-genre-classifier/` does not exist, `--classifier finetuned` logs a WARNING and automatically falls back to the zero-shot fast model — no crash, no exception.

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
