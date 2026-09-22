"""
live_stream_ingest.py - Real-time audio stream transcription & ingestion for VaniStream

Decodes live Icecast/HTTP MP3 or HLS streams into rolling PCM audio chunks via ffmpeg,
transcribes each chunk using faster-whisper with VAD, stitches overlapping text with
strip_overlap, saves audio/transcript artifacts matching existing conventions, and indexes
segments immediately into FAISS and SQLite via nlp_pipeline and vector_store.

CLI usage:
  python live_stream_ingest.py --url http://stream-url/live.mp3
  python live_stream_ingest.py --url https://example.com/live.m3u8 --chunk-seconds 30 --overlap-seconds 5 --fast
"""

import argparse
import datetime as dt
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from typing import Optional

import numpy as np

from air_bulletins_scraper import sha256_file, sleep_jitter, strip_overlap, write_csv_rows
from ingest_pipeline import _mark_processed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("live_stream_ingest")


def check_ffmpeg() -> None:
    """Verify that ffmpeg binary is present on PATH."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg was not found on your system PATH. "
            "Please install ffmpeg (e.g. 'winget install Gyan.FFmpeg' on Windows or "
            "'sudo apt install ffmpeg' on Linux/Debian) and verify it is available on PATH."
        )


def save_pcm_as_mp3(pcm_bytes: bytes, mp3_path: str, sample_rate: int = 16000) -> None:
    """Encode raw 16kHz mono 16-bit PCM bytes into an MP3 file using ffmpeg."""
    os.makedirs(os.path.dirname(mp3_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-i", "-",
        "-codec:a", "libmp3lame",
        "-b:a", "128k",
        mp3_path,
    ]
    proc = subprocess.run(
        cmd,
        input=pcm_bytes,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        logger.warning("ffmpeg MP3 encoding exited with code %d for %s", proc.returncode, mp3_path)


def get_transcriber_model(transcriber: str, model_name: str, device: str):
    """Load and cache transcriber model instance."""
    if transcriber == "faster-whisper":
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("faster-whisper is not installed") from exc
        logger.info("Loading faster-whisper model '%s' on %s (int8)...", model_name, device)
        return WhisperModel(model_name, device=device, compute_type="int8")

    if transcriber == "whisper":
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError("openai-whisper is not installed") from exc
        logger.info("Loading whisper model '%s' on %s...", model_name, device)
        return whisper.load_model(model_name, device=device)

    raise ValueError(f"Unknown transcriber: {transcriber}")


def read_exact(pipe, num_bytes: int, stop_event: Optional[threading.Event] = None) -> bytes:
    """Read exactly num_bytes from a binary pipe, breaking early if stop_event is signaled."""
    chunks: list[bytes] = []
    bytes_read = 0
    while bytes_read < num_bytes:
        if stop_event is not None and stop_event.is_set():
            break
        block_size = min(4096, num_bytes - bytes_read)
        data = pipe.read(block_size)
        if not data:
            break
        chunks.append(data)
        bytes_read += len(data)
    return b"".join(chunks)


class LiveStreamManager:
    """Manages background live stream transcription and ingestion."""

    def __init__(self):
        self._lock = threading.Lock()
        self.running: bool = False
        self.url: Optional[str] = None
        self.started_at: Optional[str] = None
        self.segments_ingested: int = 0
        self.last_chunk_at: Optional[str] = None
        self.stream_id: Optional[str] = None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._model = None
        self._model_key: Optional[tuple[str, str, str]] = None

    def get_status(self) -> dict:
        """Return status dictionary matching API requirements."""
        with self._lock:
            return {
                "running": self.running,
                "url": self.url if self.running else None,
                "started_at": self.started_at if self.running else None,
                "segments_ingested": self.segments_ingested if self.running else 0,
                "last_chunk_at": self.last_chunk_at if self.running else None,
            }

    def start(
        self,
        url: str,
        language: str = "english",
        chunk_seconds: int = 30,
        overlap_seconds: int = 5,
        fast: bool = True,
        transcriber: str = "faster-whisper",
        model: str = "base",
        device: str = "cpu",
        out_dir: str = "output",
        csv_path: str = os.path.join("output", "metadata.csv"),
        category: str = "hourly-news",
        min_delay: float = 1.0,
        max_delay: float = 3.0,
        max_retries: int = 3,
    ) -> str:
        """Start live stream transcription in a background thread."""
        check_ffmpeg()

        with self._lock:
            if self.running:
                raise RuntimeError(f"A stream is already active with ID '{self.stream_id}'. Stop it first.")

            sid = uuid.uuid4().hex[:8]
            self.stream_id = sid
            self.running = True
            self.url = url
            self.started_at = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
            self.segments_ingested = 0
            self.last_chunk_at = None
            self._stop_event.clear()

            self._thread = threading.Thread(
                target=self._run_loop,
                kwargs={
                    "url": url,
                    "language": language,
                    "chunk_seconds": chunk_seconds,
                    "overlap_seconds": overlap_seconds,
                    "fast": fast,
                    "transcriber": transcriber,
                    "model_name": model,
                    "device": device,
                    "out_dir": out_dir,
                    "csv_path": csv_path,
                    "category": category,
                    "min_delay": min_delay,
                    "max_delay": max_delay,
                    "max_retries": max_retries,
                },
                daemon=True,
            )
            self._thread.start()
            logger.info("Started live stream %s for url: %s", sid, url)
            return sid

    def stop(self, stream_id: Optional[str] = None) -> dict:
        """Stop active stream cleanly, flushing current audio buffer."""
        with self._lock:
            if not self.running:
                return {"status": "not_running", "stream_id": None}

            sid = self.stream_id
            if stream_id and stream_id != sid:
                raise ValueError(f"Stream ID mismatch: active stream is '{sid}', got '{stream_id}'")

            logger.info("Stopping live stream %s...", sid)
            self._stop_event.set()

        # Join thread with timeout to allow clean buffer flush
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=15.0)

        # Force terminate ffmpeg process if still alive
        if self._ffmpeg_proc and self._ffmpeg_proc.poll() is None:
            try:
                self._ffmpeg_proc.terminate()
                self._ffmpeg_proc.wait(timeout=3)
            except Exception:
                try:
                    self._ffmpeg_proc.kill()
                except Exception:
                    pass

        with self._lock:
            self.running = False
            return {"status": "stopped", "stream_id": sid}

    def _spawn_ffmpeg(self, url: str) -> subprocess.Popen:
        """Spawn ffmpeg subprocess to decode stream to 16kHz mono PCM stdout."""
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
        ]
        if url.startswith("http://") or url.startswith("https://"):
            cmd.extend([
                "-reconnect", "1",
                "-reconnect_at_eof", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "5",
            ])
        cmd.extend([
            "-i", url,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-f", "s16le",
            "-",
        ])
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**7,
        )

    def _get_model(self, transcriber: str, model_name: str, device: str):
        key = (transcriber, model_name, device)
        if self._model is None or self._model_key != key:
            self._model = get_transcriber_model(transcriber, model_name, device)
            self._model_key = key
        return self._model

    def _run_loop(
        self,
        url: str,
        language: str,
        chunk_seconds: int,
        overlap_seconds: int,
        fast: bool,
        transcriber: str,
        model_name: str,
        device: str,
        out_dir: str,
        csv_path: str,
        category: str,
        min_delay: float,
        max_delay: float,
        max_retries: int,
    ) -> None:
        """Main streaming ingestion loop."""
        try:
            model = self._get_model(transcriber, model_name, device)
        except Exception as e:
            logger.error("Failed to load transcriber model: %s", e)
            with self._lock:
                self.running = False
            return

        sample_rate = 16000
        bytes_per_sec = sample_rate * 2  # 16-bit mono = 2 bytes/sample
        chunk_bytes = max(1, chunk_seconds) * bytes_per_sec
        overlap_bytes = max(0, min(overlap_seconds, chunk_seconds - 1)) * bytes_per_sec
        step_bytes = chunk_bytes - overlap_bytes
        overlap_buffer = b""
        prev_text = ""
        chunk_idx = 1
        retries = 0

        # Initialize deduplication cache once at stream start
        try:
            import vector_store
            cached_embeddings = vector_store.get_all_embeddings()
            cached_texts = vector_store.get_all_texts()
            logger.info("Loaded initial dedup cache: %d existing items",
                        len(cached_texts) if cached_texts else 0)
        except Exception as e:
            logger.warning("Could not pre-load vector store cache: %s", e)
            cached_embeddings = None
            cached_texts = []

        while not self._stop_event.is_set():
            proc = self._spawn_ffmpeg(url)
            self._ffmpeg_proc = proc

            logger.info("Connected to stream %s (chunk=%ds, overlap=%ds)", url, chunk_seconds, overlap_seconds)

            while not self._stop_event.is_set():
                needed = chunk_bytes if len(overlap_buffer) == 0 else step_bytes
                raw_data = read_exact(proc.stdout, needed, self._stop_event)

                if not raw_data:
                    # Pipe closed or EOF reached
                    break

                current_pcm = overlap_buffer + raw_data

                # If audio is at least 1 second long, process it
                if len(current_pcm) >= bytes_per_sec:
                    retries = 0  # Reset retry counter on successful audio read
                    now = dt.datetime.utcnow()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H%M")
                    slug = f"Live-{language.capitalize()}-{time_str}-{chunk_idx:04d}"

                    # 1. Save audio snapshot
                    audio_dir = os.path.join(out_dir, "audio", date_str)
                    audio_path = os.path.join(audio_dir, f"{slug}.mp3")
                    save_pcm_as_mp3(current_pcm, audio_path, sample_rate=sample_rate)

                    # 2. Transcribe
                    transcript_raw = ""
                    try:
                        if transcriber == "faster-whisper":
                            segments, _info = model.transcribe(
                                audio_path,
                                vad_filter=True,
                                language=language if language != "auto" else None,
                            )
                            transcript_raw = " ".join(seg.text.strip() for seg in segments).strip()
                        elif transcriber == "whisper":
                            result = model.transcribe(
                                audio_path,
                                language=language if language != "auto" else None,
                            )
                            transcript_raw = (result.get("text") or "").strip()
                    except Exception as e:
                        logger.warning("Transcription failed for chunk %d: %s", chunk_idx, e)

                    # 3. Stitch text by removing overlap
                    transcript_text = strip_overlap(prev_text, transcript_raw, min_overlap=80)
                    prev_text = transcript_raw

                    logger.info("Chunk #%d transcribed: '%s' (raw=%d chars, cleaned=%d chars)",
                                chunk_idx, transcript_text[:60], len(transcript_raw), len(transcript_text))

                    # 4. Ingest and persist if text was transcribed
                    rel_transcript = ""
                    if transcript_text:
                        text_dir = os.path.join(out_dir, "text", date_str)
                        os.makedirs(text_dir, exist_ok=True)
                        transcript_path = os.path.join(text_dir, f"{slug}.txt")
                        with open(transcript_path, "w", encoding="utf-8") as f:
                            f.write(transcript_text)
                        rel_transcript = os.path.relpath(transcript_path, out_dir)

                    # Write metadata row
                    file_hash = sha256_file(audio_path) if os.path.isfile(audio_path) else ""
                    row = {
                        "title": f"AIR Live Stream ({date_str} {time_str}) - #{chunk_idx}",
                        "date": date_str,
                        "language": language,
                        "category": category,
                        "audio_url": url,
                        "audio_file": os.path.relpath(audio_path, out_dir),
                        "transcript_file": rel_transcript,
                        "sha256": file_hash,
                        "downloaded_at": now.isoformat(timespec="seconds") + "Z",
                    }
                    write_csv_rows(csv_path, [row])

                    # 5. Add to vector store in real-time
                    if transcript_text and rel_transcript:
                        try:
                            import nlp_pipeline
                            import vector_store
                            segments = nlp_pipeline.process_transcript(
                                transcript_text,
                                existing_embeddings=cached_embeddings,
                                existing_texts=cached_texts,
                                fast=fast,
                                mmr_threshold=0.85,
                                detailed=False,
                            )
                            if segments:
                                hour_num = now.hour
                                added = vector_store.add_segments(segments, date_str, hour_num, rel_transcript)
                                with self._lock:
                                    self.segments_ingested += added
                                logger.info("Live ingested %d segments from chunk #%d (hour=%d)", added, chunk_idx, hour_num)

                                # Incrementally update deduplication cache in memory (avoids full reload on every chunk)
                                new_embs = np.vstack([seg["embedding"] for seg in segments])
                                if cached_embeddings is None or len(cached_embeddings) == 0:
                                    cached_embeddings = new_embs
                                else:
                                    cached_embeddings = np.vstack([cached_embeddings, new_embs])
                                if cached_texts is not None:
                                    cached_texts.extend([seg["text"] for seg in segments])

                            _mark_processed(rel_transcript)
                        except Exception as e:
                            logger.error("Vector store ingestion failed for chunk #%d: %s", chunk_idx, e)

                    with self._lock:
                        self.last_chunk_at = now.isoformat(timespec="seconds") + "Z"

                    chunk_idx += 1

                # Update rolling overlap buffer
                if overlap_bytes > 0 and len(current_pcm) >= overlap_bytes:
                    overlap_buffer = current_pcm[-overlap_bytes:]
                else:
                    overlap_buffer = b""

                if self._stop_event.is_set():
                    break

            # Process clean termination
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    pass

            if self._stop_event.is_set():
                break

            # Handle unexpected stream disconnection with jittered retry
            retries += 1
            if retries > max_retries:
                logger.error("Max stream retries (%d) reached for %s. Exiting stream loop.", max_retries, url)
                break

            logger.warning("Stream connection dropped. Retry %d/%d in jittered delay...", retries, max_retries)
            sleep_jitter(min_delay, max_delay)

        with self._lock:
            self.running = False
            logger.info("Live stream loop ended (segments_ingested=%d)", self.segments_ingested)


# Global manager instance for application use
live_manager = LiveStreamManager()


def main() -> int:
    """CLI entry point for standalone live stream ingestion."""
    parser = argparse.ArgumentParser(description="Live stream audio transcription & ingestion.")
    parser.add_argument("--url", required=True, help="Live stream URL (Icecast MP3, HLS .m3u8, or HTTP stream)")
    parser.add_argument("--language", default="english", help="Spoken language (e.g. english, hindi, urdu)")
    parser.add_argument("--chunk-seconds", type=int, default=30, help="Rolling audio window duration in seconds")
    parser.add_argument("--overlap-seconds", type=int, default=5, help="Audio window overlap in seconds")
    parser.add_argument("--fast", action="store_true", help="Use lighter zero-shot classifier model")
    parser.add_argument("--transcriber", default="faster-whisper", choices=["faster-whisper", "whisper"],
                        help="ASR transcriber engine")
    parser.add_argument("--model", default="base", help="Whisper model name (tiny, base, small, medium, large)")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--out-dir", default="output", help="Base output directory")
    parser.add_argument("--csv", default=os.path.join("output", "metadata.csv"), help="Metadata CSV path")
    parser.add_argument("--category", default="hourly-news", help="Category label for live bulletins")
    parser.add_argument("--min-delay", type=float, default=1.0, help="Min retry delay seconds")
    parser.add_argument("--max-delay", type=float, default=3.0, help="Max retry delay seconds")
    parser.add_argument("--max-retries", type=int, default=3, help="Max stream reconnect retries")
    args = parser.parse_args()

    try:
        check_ffmpeg()
    except RuntimeError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print(f"Connecting to live stream: {args.url}")
    print(f"Chunk size: {args.chunk_seconds}s (overlap: {args.overlap_seconds}s) | Engine: {args.transcriber} ({args.model})")
    print("Press Ctrl+C to stop stream cleanly.\n")

    manager = LiveStreamManager()
    try:
        sid = manager.start(
            url=args.url,
            language=args.language,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            fast=args.fast,
            transcriber=args.transcriber,
            model=args.model,
            device=args.device,
            out_dir=args.out_dir,
            csv_path=args.csv,
            category=args.category,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
            max_retries=args.max_retries,
        )
        while manager.running:
            if manager._thread:
                manager._thread.join(timeout=1.0)
    except KeyboardInterrupt:
        print("\nInterrupt received. Flushing audio buffer and halting stream...")
        manager.stop()
    except Exception as exc:
        print(f"Fatal error in stream ingest: {exc}", file=sys.stderr)
        manager.stop()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
