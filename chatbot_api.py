"""
chatbot_api.py - FastAPI server for the AIR News Intelligence chatbot

Endpoints:
  POST /chat    - Send a message and get RAG-powered response
  GET  /stats   - Vector store statistics
  GET  /health  - Health check
  GET  /        - Serve the web chat UI

Run with:
  uvicorn chatbot_api:app --reload --port 8000
"""

import json
import logging
import os
import uuid
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)
logger = logging.getLogger("chatbot_api")

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from live_stream_ingest import live_manager
import rag_engine
import vector_store

app = FastAPI(
    title="VaniStream - AIR News Intelligence",
    description="RAG-powered chatbot for All India Radio broadcast news",
    version="1.0.0",
)

# In-memory conversation history: session_id -> list of {role, content}
_sessions: dict[str, list[dict]] = {}


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Chat request body."""
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response body."""
    reply: str
    sources: list[dict]
    filters: dict
    session_id: str


class StreamStartRequest(BaseModel):
    """Live stream start request."""
    url: str
    language: str = "english"
    chunk_seconds: int = 30
    overlap_seconds: int = 5
    fast: bool = True
    transcriber: str = "faster-whisper"
    model: str = "base"


class StreamStopRequest(BaseModel):
    """Live stream stop request."""
    stream_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Process a chat message through the RAG engine."""
    sid = req.session_id or str(uuid.uuid4())

    if sid not in _sessions:
        _sessions[sid] = []

    # Prior turns for context (exclude current message)
    history = _sessions[sid][-6:]  # 3 turns = 6 messages (user+model)

    # Add user message to history
    _sessions[sid].append({"role": "user", "content": req.message})

    result = rag_engine.answer_query(req.message, history=history)

    # Add assistant response to history
    _sessions[sid].append({"role": "model", "content": result["answer"]})

    # Trim history to avoid unbounded growth
    if len(_sessions[sid]) > 20:
        _sessions[sid] = _sessions[sid][-20:]

    return ChatResponse(
        reply=result["answer"],
        sources=result["sources"],
        filters=result["filters_applied"],
        session_id=sid,
    )


@app.get("/stats")
async def stats():
    """Return vector store statistics."""
    return vector_store.get_stats()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "generation_mode": rag_engine.get_generation_mode()}


@app.get("/dedup")
async def dedup():
    """Return deduplication report statistics and top dropped duplicate pairs."""
    report_path = "dedup_report.json"
    if not os.path.isfile(report_path):
        return {"error": "no report"}
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


@app.get("/timeline")
async def timeline(q: str = "", date_from: Optional[str] = None, date_to: Optional[str] = None):
    """Retrieve chronologically grouped segments matching a query."""
    if not q.strip():
        return {"groups": [], "filters_applied": {}}
    return rag_engine.timeline_query(q, date_from=date_from, date_to=date_to)


@app.get("/audio/{faiss_id:int}")
async def get_audio(faiss_id: int):
    """Serve the audio bulletin MP3 corresponding to a segment's faiss_id."""
    db_path = vector_store.SQLITE_DB_PATH
    if not os.path.isfile(db_path):
        return JSONResponse(status_code=404, content={"error": "Database not initialized"})

    db = vector_store._get_db(db_path)
    row = db.execute("SELECT source_file FROM segments WHERE faiss_id = ?", (faiss_id,)).fetchone()
    db.close()

    if not row:
        return JSONResponse(status_code=404, content={"error": f"Segment {faiss_id} not found"})

    source_file = row["source_file"]
    audio_relpath, exists = vector_store.resolve_audio_path(source_file)

    # Security check: ensure path stays strictly inside output/audio
    audio_dir = os.path.abspath(os.path.join("output", "audio"))
    target_abs = os.path.abspath(audio_relpath)
    try:
        if os.path.commonpath([audio_dir, target_abs]) != audio_dir:
            return JSONResponse(status_code=403, content={"error": "Access denied: path escapes audio directory"})
    except ValueError:
        return JSONResponse(status_code=403, content={"error": "Access denied: path escapes audio directory"})

    if not exists or not os.path.isfile(target_abs):
        return JSONResponse(status_code=404, content={"error": f"Audio file not found on disk for segment {faiss_id}"})

    # Detect actual audio format from file header
    with open(target_abs, "rb") as f:
        header = f.read(12)
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        mime = "audio/wav"
    else:
        mime = "audio/mpeg"

    return FileResponse(
        target_abs,
        media_type=mime,
        filename=os.path.basename(target_abs),
    )


@app.post("/stream/start")
async def stream_start(req: StreamStartRequest):
    """Launch real-time radio stream transcription loop in background thread."""
    status = live_manager.get_status()
    if status["running"]:
        return JSONResponse(
            status_code=400,
            content={"error": f"A stream is already active with ID '{live_manager.stream_id}'. Stop it first."}
        )
    try:
        sid = live_manager.start(
            url=req.url,
            language=req.language,
            chunk_seconds=req.chunk_seconds,
            overlap_seconds=req.overlap_seconds,
            fast=req.fast,
            transcriber=req.transcriber,
            model=req.model,
        )
        return {"stream_id": sid, "status": "started", "url": req.url}
    except RuntimeError as err:
        return JSONResponse(status_code=400, content={"error": str(err)})
    except Exception as err:
        logger.error("Failed to start stream: %s", err)
        return JSONResponse(status_code=500, content={"error": str(err)})


@app.post("/stream/stop")
async def stream_stop(req: Optional[StreamStopRequest] = None):
    """Cleanly stop the active live stream, flushing the current audio buffer."""
    sid = req.stream_id if req else None
    try:
        res = live_manager.stop(stream_id=sid)
        return res
    except ValueError as err:
        return JSONResponse(status_code=400, content={"error": str(err)})
    except Exception as err:
        logger.error("Failed to stop stream: %s", err)
        return JSONResponse(status_code=500, content={"error": str(err)})


@app.get("/stream/status")
async def stream_status():
    """Return status of live stream transcription."""
    return live_manager.get_status()


@app.get("/", response_class=HTMLResponse)
async def ui():
    """Serve the chatbot web UI."""
    return CHAT_HTML


# ---------------------------------------------------------------------------
# Inline Chat UI
# ---------------------------------------------------------------------------

CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VaniStream - AIR News Intelligence</title>
<meta name="description" content="AI-powered chatbot for All India Radio broadcast news intelligence">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #21262d;
    --surface: #1c2333;
    --border: #30363d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #6e7681;
    --accent: #58a6ff;
    --accent-glow: rgba(88, 166, 255, 0.15);
    --accent-gradient: linear-gradient(135deg, #58a6ff 0%, #bc8cff 100%);
    --user-bubble: #1f3a5f;
    --bot-bubble: #1c2333;
    --success: #3fb950;
    --warning: #d29922;
    --radius: 12px;
    --shadow: 0 8px 32px rgba(0,0,0,0.4);
  }

  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* Header */
  .header {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
  }
  .header-logo {
    width: 40px; height: 40px;
    background: var(--accent-gradient);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; font-weight: 700; color: #fff;
  }
  .header h1 {
    font-size: 20px; font-weight: 700;
    background: var(--accent-gradient);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .header p { font-size: 12px; color: var(--text-secondary); }
  .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 8px var(--success);
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .live-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    color: var(--text-muted);
    transition: all 0.3s ease;
    user-select: none;
  }
  .live-pill.active {
    background: rgba(63, 185, 80, 0.15);
    border-color: var(--success);
    color: var(--success);
    box-shadow: 0 0 10px rgba(63, 185, 80, 0.3);
    animation: livePillPulse 2s infinite ease-in-out;
  }
  .live-pill-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
  }
  @keyframes livePillPulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.75; transform: scale(0.97); }
  }

  /* Chat area */
  .chat-container {
    flex: 1; overflow-y: auto; padding: 24px;
    display: flex; flex-direction: column; gap: 16px;
    scroll-behavior: smooth;
  }
  .chat-container::-webkit-scrollbar { width: 6px; }
  .chat-container::-webkit-scrollbar-track { background: transparent; }
  .chat-container::-webkit-scrollbar-thumb {
    background: var(--border); border-radius: 3px;
  }

  .message { max-width: 75%; animation: fadeIn 0.3s ease; }
  .message.user { align-self: flex-end; }
  .message.bot { align-self: flex-start; }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .bubble {
    padding: 14px 18px; border-radius: var(--radius);
    font-size: 14px; line-height: 1.6; word-wrap: break-word;
  }
  .user .bubble {
    background: var(--user-bubble);
    border: 1px solid rgba(88,166,255,0.2);
    border-bottom-right-radius: 4px;
  }
  .bot .bubble {
    background: var(--bot-bubble);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }
  .msg-label {
    font-size: 11px; color: var(--text-muted);
    margin-bottom: 4px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .user .msg-label { text-align: right; }

  /* Sources */
  .sources {
    margin-top: 10px; padding-top: 10px;
    border-top: 1px solid var(--border);
  }
  .sources-title {
    font-size: 11px; color: var(--text-muted); font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;
  }
  .source-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--bg-tertiary); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 10px; margin: 3px 4px 3px 0;
    font-size: 11px; color: var(--text-secondary);
    transition: border-color 0.2s;
  }
  .source-chip:hover { border-color: var(--accent); }
  .source-genre {
    background: var(--accent-glow); color: var(--accent);
    padding: 1px 6px; border-radius: 4px; font-weight: 600;
  }

  /* Typing indicator */
  .typing { display: flex; gap: 4px; padding: 8px 0; }
  .typing span {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--text-muted);
    animation: bounce 1.4s infinite;
  }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-8px); }
  }

  /* Input area */
  .input-area {
    background: var(--bg-secondary);
    border-top: 1px solid var(--border);
    padding: 16px 24px; flex-shrink: 0;
  }
  .input-row {
    display: flex; gap: 12px; max-width: 900px; margin: 0 auto;
  }
  #chatInput {
    flex: 1; background: var(--bg-tertiary);
    border: 1px solid var(--border); border-radius: var(--radius);
    padding: 14px 18px; color: var(--text-primary);
    font-family: inherit; font-size: 14px; outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
    resize: none; min-height: 48px; max-height: 120px;
  }
  #chatInput:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
  }
  #chatInput::placeholder { color: var(--text-muted); }
  #sendBtn {
    background: var(--accent-gradient);
    border: none; border-radius: var(--radius);
    padding: 0 24px; color: #fff; font-weight: 600;
    font-size: 14px; cursor: pointer;
    transition: transform 0.15s, box-shadow 0.2s;
    white-space: nowrap;
  }
  #sendBtn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(88,166,255,0.3);
  }
  #sendBtn:active { transform: translateY(0); }
  #sendBtn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
  #timelineBtn {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0 16px;
    color: var(--accent);
    font-weight: 600;
    font-size: 13px;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.2s;
  }
  #timelineBtn:hover {
    border-color: var(--accent);
    background: var(--accent-glow);
    transform: translateY(-1px);
  }
  #timelineBtn:active { transform: translateY(0); }
  .audio-btn {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--accent);
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    cursor: pointer;
    transition: all 0.2s ease;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-left: 4px;
  }
  .audio-btn:hover {
    border-color: var(--accent);
    background: var(--accent-glow);
  }
  .audio-btn.playing {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
    box-shadow: 0 0 8px var(--accent-glow);
  }
  #audioPlayerBar {
    display: none;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border);
    padding: 8px 24px;
    align-items: center;
    justify-content: space-between;
    font-size: 12px;
    flex-shrink: 0;
  }
  #audioPlayerBar.active {
    display: flex;
  }
  #sharedAudioPlayer {
    height: 32px;
    outline: none;
  }

  /* Welcome */
  .welcome {
    text-align: center; margin: auto; padding: 40px;
    max-width: 500px;
  }
  .welcome h2 {
    font-size: 24px; margin-bottom: 8px;
    background: var(--accent-gradient);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .welcome p { color: var(--text-secondary); font-size: 14px; line-height: 1.6; }
  .suggestions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 20px; justify-content: center; }
  .suggestion {
    background: var(--bg-tertiary); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 14px; font-size: 13px;
    color: var(--text-secondary); cursor: pointer;
    transition: all 0.2s;
  }
  .suggestion:hover {
    border-color: var(--accent); color: var(--accent);
    background: var(--accent-glow);
  }

  @media (max-width: 600px) {
    .message { max-width: 90%; }
    .header { padding: 12px 16px; }
    .input-area { padding: 12px 16px; }
  }
</style>
</head>
<body>

<div class="header">
  <div class="header-logo">V</div>
  <div style="flex: 1;">
    <h1>VaniStream</h1>
    <p>AIR News Intelligence Assistant</p>
    <div id="dedupBanner" style="display:none; margin-top: 4px; font-size: 11px;">
      <span id="dedupStatsText" style="color: var(--success); font-weight: 500;"></span>
      <details id="dedupDetails" style="margin-top: 4px; cursor: pointer;">
        <summary style="color: var(--accent); font-size: 11px; outline: none; user-select: none;">View deduplication examples</summary>
        <div id="dedupExamples" style="margin-top: 6px; padding: 8px 12px; background: var(--bg-tertiary); border: 1px solid var(--border); border-radius: 6px; font-size: 11px; line-height: 1.4;"></div>
      </details>
    </div>
  </div>
  <div style="display: flex; align-items: center; gap: 10px; margin-left: auto;">
    <div id="livePill" class="live-pill" title="Live stream: Idle">
      <span class="live-pill-dot"></span>LIVE
    </div>
    <div class="status-dot" title="System online"></div>
  </div>
</div>

<div class="chat-container" id="chatBox">
  <div class="welcome" id="welcome">
    <h2>Welcome to VaniStream</h2>
    <p>Ask me anything about recent All India Radio news broadcasts.
       I can search by topic, date, and genre.</p>
    <div class="suggestions">
      <div class="suggestion" onclick="askSuggestion(this)">What happened in politics last week?</div>
      <div class="suggestion" onclick="askSuggestion(this)">Latest sports news</div>
      <div class="suggestion" onclick="askSuggestion(this)">Economy updates today</div>
      <div class="suggestion" onclick="askSuggestion(this)">Defence news last 15 days</div>
    </div>
  </div>
</div>

<div id="audioPlayerBar">
  <div style="display: flex; align-items: center; gap: 10px; flex: 1; min-width: 0; margin-right: 16px;">
    <span id="playerIcon" style="color: var(--accent); font-size: 13px;">▶</span>
    <div style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
      <span id="playerTrackInfo" style="font-weight: 600; color: var(--text-primary);">Broadcasting</span>
      <span style="color: var(--text-muted); font-size: 11px; margin-left: 8px;">(plays full bulletin — sentence-level alignment is future work)</span>
    </div>
  </div>
  <audio id="sharedAudioPlayer" controls></audio>
  <button id="closePlayerBtn" onclick="stopAudio()" title="Close audio player" style="background: none; border: none; color: var(--text-muted); font-size: 18px; cursor: pointer; padding: 0 8px; margin-left: 8px; line-height: 1;">✕</button>
</div>

<div class="input-area">
  <div class="input-row">
    <textarea id="chatInput" placeholder="Ask about AIR news broadcasts..."
              rows="1" onkeydown="handleKey(event)"></textarea>
    <button id="timelineBtn" onclick="showTimeline()" title="View story evolution across dates/hours">Timeline</button>
    <button id="sendBtn" onclick="sendMessage()">Send</button>
  </div>
</div>

<script>
let sessionId = null;

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function askSuggestion(el) {
  document.getElementById('chatInput').value = el.textContent;
  sendMessage();
}

function addMessage(role, text, sources) {
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();

  const box = document.getElementById('chatBox');
  const div = document.createElement('div');
  div.className = 'message ' + role;

  let html = '<div class="msg-label">' + (role === 'user' ? 'You' : 'VaniStream') + '</div>';
  html += '<div class="bubble">' + escapeHtml(text).replace(/\\n/g, '<br>') + '</div>';

  if (sources && sources.length > 0) {
    html += '<div class="sources"><div class="sources-title">Sources</div>';
    sources.forEach(s => {
      let audioBtn = '';
      if (s.has_audio && s.faiss_id !== undefined) {
        const lbl = escapeHtml(s.date + ' ' + (s.hour ? s.hour : ''));
        audioBtn = '<button class="audio-btn" data-faiss-id="' + s.faiss_id + '" data-label="' + lbl + '" onclick="playAudioFromBtn(this)">▶ Audio</button>';
      }
      html += '<div class="source-chip">'
        + '<span>' + s.date + '</span>'
        + '<span class="source-genre">' + s.genre + '</span>'
        + '<span>score: ' + (s.score ? s.score.toFixed(2) : '-') + '</span>'
        + audioBtn
        + '</div>';
    });
    html += '</div>';
  }
  div.innerHTML = html;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function showTyping() {
  const box = document.getElementById('chatBox');
  const div = document.createElement('div');
  div.className = 'message bot';
  div.id = 'typingIndicator';
  div.innerHTML = '<div class="msg-label">VaniStream</div>'
    + '<div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>';
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function removeTyping() {
  const el = document.getElementById('typingIndicator');
  if (el) el.remove();
}

function escapeHtml(t) {
  const d = document.createElement('div');
  d.textContent = t;
  return d.innerHTML;
}

async function sendMessage() {
  const input = document.getElementById('chatInput');
  const btn = document.getElementById('sendBtn');
  const msg = input.value.trim();
  if (!msg) return;

  input.value = '';
  input.style.height = 'auto';
  btn.disabled = true;

  addMessage('user', msg);
  showTyping();

  try {
    const body = { message: msg };
    if (sessionId) body.session_id = sessionId;

    const resp = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    sessionId = data.session_id;
    removeTyping();
    addMessage('bot', data.reply, data.sources);
  } catch (err) {
    removeTyping();
    addMessage('bot', 'Sorry, something went wrong. Please try again.');
  }
  btn.disabled = false;
  input.focus();
}

function addTimelineMessage(query, data) {
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();

  const box = document.getElementById('chatBox');
  const div = document.createElement('div');
  div.className = 'message bot';

  let html = '<div class="msg-label">VaniStream &bull; Timeline Analysis</div>';
  html += '<div class="bubble">';
  html += '<div style="font-weight: 600; color: var(--accent); margin-bottom: 10px; font-size: 14px;">Chronological Timeline: "' + escapeHtml(query) + '"</div>';

  if (!data.groups || data.groups.length === 0) {
    html += '<div style="color: var(--text-secondary);">No timeline segments found for this query in the current corpus.</div>';
  } else {
    data.groups.forEach(g => {
      html += '<div style="margin-top: 12px; margin-bottom: 6px; font-weight: 700; color: var(--text-primary); font-size: 13px; display: flex; align-items: center; gap: 8px;">';
      html += '<span style="background: var(--accent); width: 8px; height: 8px; border-radius: 50%; display: inline-block;"></span>';
      html += '<span>' + escapeHtml(g.date) + '</span>';
      html += '</div>';

      g.segments.forEach(seg => {
        const hourStr = String(seg.hour).padStart(4, '0');
        const formattedHour = hourStr.slice(0, 2) + ':' + hourStr.slice(2);
        let audioBtn = '';
        if (seg.has_audio && seg.faiss_id !== undefined) {
          const lbl = escapeHtml(seg.date + ' ' + formattedHour);
          audioBtn = '<button class="audio-btn" data-faiss-id="' + seg.faiss_id + '" data-label="' + lbl + '" onclick="playAudioFromBtn(this)">▶ Audio</button>';
        }
        html += '<div style="margin-left: 12px; margin-bottom: 8px; padding: 10px 14px; background: var(--bg-tertiary); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 6px;">';
        html += '<div style="display: flex; gap: 8px; margin-bottom: 4px; align-items: center; font-size: 11px;">';
        html += '<span style="font-weight: 600; color: var(--text-primary);">' + formattedHour + ' hrs</span>';
        html += '<span class="source-genre">' + escapeHtml(seg.genre) + '</span>';
        if (seg.score) html += '<span style="color: var(--text-muted);">relevance: ' + seg.score.toFixed(2) + '</span>';
        html += audioBtn;
        html += '</div>';
        html += '<div style="color: var(--text-primary); font-size: 13px; line-height: 1.5;">' + escapeHtml(seg.text) + '</div>';
        html += '</div>';
      });
    });
  }

  html += '</div>';
  div.innerHTML = html;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

async function showTimeline() {
  const input = document.getElementById('chatInput');
  let q = input.value.trim();
  if (!q) {
    q = "news";
  }

  showTyping();
  try {
    const res = await fetch('/timeline?q=' + encodeURIComponent(q));
    const data = await res.json();
    removeTyping();
    addTimelineMessage(q, data);
  } catch (err) {
    removeTyping();
    addMessage('bot', 'Failed to fetch timeline. Please try again.');
  }
}

// Auto-resize textarea
document.getElementById('chatInput').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

let currentPlayingFaissId = null;
let currentPlayingBtn = null;

function stopAudio() {
  const player = document.getElementById('sharedAudioPlayer');
  const bar = document.getElementById('audioPlayerBar');
  if (player) {
    player.pause();
    player.currentTime = 0;
  }
  if (currentPlayingBtn) {
    currentPlayingBtn.classList.remove('playing');
    currentPlayingBtn.innerHTML = '▶ Audio';
    currentPlayingBtn = null;
  }
  currentPlayingFaissId = null;
  if (bar) bar.classList.remove('active');
}

function playAudioFromBtn(btn) {
  const faissId = parseInt(btn.getAttribute('data-faiss-id'), 10);
  const label = btn.getAttribute('data-label') || '';
  toggleAudio(btn, faissId, label);
}

function toggleAudio(btn, faissId, label) {
  const player = document.getElementById('sharedAudioPlayer');
  const bar = document.getElementById('audioPlayerBar');
  const trackInfo = document.getElementById('playerTrackInfo');

  if (currentPlayingFaissId === faissId && !player.paused) {
    // Pause current
    player.pause();
    btn.classList.remove('playing');
    btn.innerHTML = '▶ Audio';
    currentPlayingFaissId = null;
    currentPlayingBtn = null;
    return;
  }

  // If another track was playing, reset its button
  if (currentPlayingBtn && currentPlayingBtn !== btn) {
    currentPlayingBtn.classList.remove('playing');
    currentPlayingBtn.innerHTML = '▶ Audio';
  }

  currentPlayingFaissId = faissId;
  currentPlayingBtn = btn;
  btn.classList.add('playing');
  btn.innerHTML = '⏸ Stop';

  if (trackInfo) {
    trackInfo.textContent = 'AIR Bulletin (' + label + ')';
  }
  if (bar) bar.classList.add('active');

  player.src = '/audio/' + faissId;
  player.play().catch(e => {
    console.error('Audio playback failed', e);
    stopAudio();
  });
}

// Attach player event listeners
window.addEventListener('DOMContentLoaded', () => {
  const player = document.getElementById('sharedAudioPlayer');
  if (player) {
    player.addEventListener('ended', stopAudio);
    player.addEventListener('pause', () => {
      if (player.currentTime === 0 || player.ended) {
        stopAudio();
      } else if (currentPlayingBtn) {
        currentPlayingBtn.classList.remove('playing');
        currentPlayingBtn.innerHTML = '▶ Audio';
      }
    });
    player.addEventListener('play', () => {
      if (currentPlayingBtn) {
        currentPlayingBtn.classList.add('playing');
        currentPlayingBtn.innerHTML = '⏸ Stop';
      }
    });
  }
});

async function loadDedupStats() {
  try {
    const res = await fetch('/dedup');
    const data = await res.json();
    if (data && data.totals && data.totals.total_sentences > 0) {
      const banner = document.getElementById('dedupBanner');
      const statsText = document.getElementById('dedupStatsText');
      const examplesDiv = document.getElementById('dedupExamples');
      const t = data.totals;
      statsText.textContent = `Redundancy removed: ${t.total_sentences} → ${t.novel_sentences} sentences, ${t.reduction_pct}%`;

      if (data.top_rejected_pairs && data.top_rejected_pairs.length > 0) {
        const top3 = data.top_rejected_pairs.slice(0, 3);
        let html = '';
        top3.forEach((p, idx) => {
          html += `<div style="margin-bottom: ${idx < top3.length - 1 ? '8px' : '0'}; border-bottom: ${idx < top3.length - 1 ? '1px solid var(--border)' : 'none'}; padding-bottom: ${idx < top3.length - 1 ? '6px' : '0'};">`
            + `<div style="color: var(--warning); font-weight: 600;">Dropped duplicate (sim: ${p.cosine_score}):</div>`
            + `<div style="color: var(--text-primary); margin: 2px 0;">"${escapeHtml(p.rejected_text)}"</div>`
            + `<div style="color: var(--text-muted); font-size: 10px;">Matched: "${escapeHtml(p.matched_text)}"</div>`
            + `</div>`;
        });
        examplesDiv.innerHTML = html;
      }
      banner.style.display = 'block';
    }
  } catch (e) {
    console.error('Failed to load dedup stats', e);
  }
}
loadDedupStats();

async function checkStreamStatus() {
  try {
    const res = await fetch('/stream/status');
    const data = await res.json();
    const pill = document.getElementById('livePill');
    if (!pill) return;
    if (data && data.running) {
      pill.classList.add('active');
      pill.title = `Live streaming active\nURL: ${data.url || 'Active stream'}\nSegments ingested: ${data.segments_ingested}`;
    } else {
      pill.classList.remove('active');
      pill.title = 'Live stream: Idle';
    }
  } catch (err) {
    console.debug('Stream status check failed', err);
  }
}
setInterval(checkStreamStatus, 5000);
checkStreamStatus();
</script>
</body>
</html>"""
