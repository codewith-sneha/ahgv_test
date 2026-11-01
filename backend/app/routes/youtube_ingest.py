"""
YouTube ingestion route — POST /api/ingest/youtube

Flow:
  1. Try youtube-transcript-api (free, no download needed)
  2. If captions disabled → yt-dlp audio download + Groq Whisper transcription
"""
from __future__ import annotations

import os
import uuid
import tempfile
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from qdrant_client.http.models import PointStruct

from app.config import COLLECTION_NAME
from app.services.qdrant_service import qdrant_client
from fastembed import TextEmbedding

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _embed_model

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

log = logging.getLogger("youtube_ingest")
router = APIRouter(prefix="/api", tags=["YouTube Ingestion"])

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embed_model


class YouTubeIngestRequest(BaseModel):
    video_id: str
    lecture_id: str
    course_id: str = ""


class YouTubeIngestResponse(BaseModel):
    status: str
    method: str
    chunks_ingested: int
    video_id: str


def _fetch_youtube_transcript(video_id: str) -> list[dict]:
    from youtube_transcript_api import YouTubeTranscriptApi
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    chunks = []
    for entry in transcript:
        chunks.append({
            "text": entry["text"].strip(),
            "start_sec": float(entry["start"]),
            "end_sec": float(entry["start"]) + float(entry["duration"]),
            "source_type": "video",
        })
    return chunks


def _transcribe_via_whisper(video_id: str) -> list[dict]:
    from groq import Groq
    import subprocess

    groq_client = Groq(api_key=os.getenv("INTERVIEW_GROQ_KEY"))

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, f"{video_id}.mp3")
        result = subprocess.run(
            ["yt-dlp", "-x", "--audio-format", "mp3", "-o", audio_path,
             f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr}")

        with open(audio_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(f"{video_id}.mp3", f.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

    chunks = []
    for seg in transcription.segments:
        chunks.append({
            "text": seg["text"].strip(),
            "start_sec": float(seg["start"]),
            "end_sec": float(seg["end"]),
            "source_type": "video",
        })
    return chunks


@router.post("/ingest/youtube", response_model=YouTubeIngestResponse)
async def ingest_youtube(body: YouTubeIngestRequest):
    video_id = body.video_id.strip()
    lecture_id = body.lecture_id.strip() or f"yt_{video_id}"

    method = "transcript"
    try:
        chunks = _fetch_youtube_transcript(video_id)
    except Exception as e:
        log.warning("Transcript unavailable for %s (%s) — falling back to Whisper", video_id, e)
        method = "whisper"
        try:
            chunks = _transcribe_via_whisper(video_id)
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Both transcript and Whisper failed: {e2}")

    if not chunks:
        raise HTTPException(status_code=422, detail="No text could be extracted from this video.")

    for c in chunks:
        c["source"] = lecture_id
        c["kind"] = "video"
        if body.course_id:
            c["course_id"] = body.course_id

    texts = [c["text"] for c in chunks]
    embeddings = [list(e) for e in get_embed_model().embed(texts)]
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=embeddings[i], payload=chunks[i])
        for i in range(len(chunks))
    ]

    try:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant upsert failed: {e}")

    return YouTubeIngestResponse(
        status="success",
        method=method,
        chunks_ingested=len(points),
        video_id=video_id,
    )