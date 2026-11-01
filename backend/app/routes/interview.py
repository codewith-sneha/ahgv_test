"""
AI Mock Interview System — POST /api/interview/start  +  POST /api/interview/respond  +  POST /api/interview/finish
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel, Field

load_dotenv()
log = logging.getLogger("interview")

router = APIRouter(prefix="/interview", tags=["Interview"])

# ── Gemini + Groq LLM with fallback ─────────────────────────────────────────

_gemini_client = None
_gemini_lock = threading.Lock()
_groq_client = None
_groq_lock = threading.Lock()

def _get_gemini_client():
    global _gemini_client
    with _gemini_lock:
        if _gemini_client is None:
            from google import genai
            api_key = os.getenv("INTERVIEW_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
            _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client

def _get_groq_client():
    global _groq_client
    with _groq_lock:
        if _groq_client is None:
            from groq import Groq
            _groq_client = Groq(api_key=os.getenv("INTERVIEW_GROQ_KEY"))  # reads INTERVIEW_GROQ_KEY
    return _groq_client

def _llm_groq(system: str, prompt: str, json_mode: bool = False, max_tokens: int = 800) -> str:
    client = _get_groq_client()
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""

def _llm(system: str, prompt: str, json_mode: bool = False, max_tokens: int = 800) -> str:
    from google.genai import types
    from google.genai.errors import ClientError
    primary = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    fallbacks_str = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite")
    gemini_models = [primary] + [m.strip() for m in fallbacks_str.split(",") if m.strip()]
    client = _get_gemini_client()
    kwargs = dict(system_instruction=system, temperature=0.4, max_output_tokens=max_tokens)
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    last_err = None
    for model in gemini_models:
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(**kwargs)
            )
            return resp.text or ""
        except ClientError as e:
            log.warning("Gemini model %s failed: %s", model, e)
            last_err = e
    # All Gemini models failed — try Groq if available
    if os.getenv("INTERVIEW_GROQ_KEY"):
        log.warning("All Gemini models failed, falling back to Groq: %s", last_err)
        try:
            return _llm_groq(system, prompt, json_mode=json_mode, max_tokens=max_tokens)
        except Exception as e:
            log.exception("Groq fallback also failed: %s", e)
            raise RuntimeError(f"Gemini and Groq both failed: {e}") from e
    raise last_err

def _parse_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except ValueError:
                pass
    return None


# ── Pydantic models ───────────────────────────────────────────────────────────

class InterviewConfig(BaseModel):
    course: str = "Data Structures and Algorithms"
    skill_level: Literal["beginner", "intermediate", "advanced"] = "intermediate"
    interview_type: Literal["technical", "conceptual", "mixed"] = "mixed"
    num_questions: int = Field(5, ge=3, le=10)
    candidate_name: str = "Student"

class InterviewTurn(BaseModel):
    question: str
    answer: str

class StartOut(BaseModel):
    session_id: str
    first_question: str

class RespondIn(BaseModel):
    session_id: str
    history: list[InterviewTurn]
    latest_answer: str
    config: InterviewConfig
    questions_asked: int

class RespondOut(BaseModel):
    next_question: Optional[str] = None
    done: bool = False

class FinishIn(BaseModel):
    history: list[InterviewTurn]
    config: InterviewConfig

class ScoreCard(BaseModel):
    overall_score: int
    technical_knowledge: int
    communication: int
    problem_solving: int
    answer_structure: int
    strengths: list[str]
    improvements: list[str]
    recommendations: list[str]
    cost_estimate: dict
    summary: str

class FinishOut(BaseModel):
    scorecard: ScoreCard


# ── System prompts ────────────────────────────────────────────────────────────

_INTERVIEWER_SYSTEM = """You are an expert technical interviewer conducting a mock interview.

STRICT OUTPUT RULE: Your response must contain ONLY the interview question itself — nothing else.
- No meta-commentary like "format: ONLY the question" or "Intermediate level?"
- No preamble like "Great answer!" or "Let's begin."
- No explanations or notes after the question.
- Just the question, ending with a question mark.

Interview rules:
1. Ask ONE focused question at a time.
2. Base each follow-up STRICTLY on the candidate's previous answer.
3. Probe depth: if the answer is vague, ask for specifics. If it's good, go deeper.
4. Stay in topic scope: {course}, {interview_type} interview, {skill_level} level.
5. Be professional but encouraging. Do NOT reveal correct answers."""

_SCORER_SYSTEM = """You are an expert interview evaluator. Analyse the full interview transcript and return a detailed scorecard as JSON.
Score each dimension 0-100 based on demonstrated understanding, depth, clarity and reasoning.
Be honest and specific — vague praise is not useful.
Return exactly this structure (no markdown, no extra keys):
{
  "overall_score": <int 0-100>,
  "technical_knowledge": <int 0-100>,
  "communication": <int 0-100>,
  "problem_solving": <int 0-100>,
  "answer_structure": <int 0-100>,
  "strengths": ["<specific strength 1>", "<specific strength 2>", "<specific strength 3>"],
  "improvements": ["<specific gap 1>", "<specific gap 2>", "<specific gap 3>"],
  "recommendations": ["<actionable prep tip 1>", "<actionable prep tip 2>", "<actionable prep tip 3>"],
  "summary": "<2-3 sentence honest overall assessment>"
}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

import uuid

def _opening_question(cfg: InterviewConfig) -> str:
    system = _INTERVIEWER_SYSTEM.format(
        course=cfg.course,
        interview_type=cfg.interview_type,
        skill_level=cfg.skill_level,
    )
    prompt = (
        f"You are starting an interview with {cfg.candidate_name}.\n"
        f"Course: {cfg.course}\nLevel: {cfg.skill_level}\nType: {cfg.interview_type}\n\n"
        "Ask the opening question. It should be broad enough to let the candidate show their understanding "
        "(e.g. 'Walk me through how a stack works and where you'd use one')."
    )
    q = _llm(system, prompt, max_tokens=400).strip()
    lines = [l for l in q.splitlines() if not l.lower().startswith("format") and "?" not in l or l.endswith("?")]
    return "\n".join(lines).strip() or q


def _follow_up(cfg: InterviewConfig, history: list[InterviewTurn], latest_answer: str) -> str:
    system = _INTERVIEWER_SYSTEM.format(
        course=cfg.course,
        interview_type=cfg.interview_type,
        skill_level=cfg.skill_level,
    )
    transcript = "\n".join(
        f"Q{i+1}: {t.question}\nA{i+1}: {t.answer}" for i, t in enumerate(history)
    )
    prompt = (
        f"Course: {cfg.course} | Level: {cfg.skill_level} | Type: {cfg.interview_type}\n\n"
        f"Transcript so far:\n{transcript}\n\n"
        f"Candidate just answered:\n{latest_answer}\n\n"
        "Ask ONE follow-up question that digs deeper into their last answer or naturally advances the topic."
    )
    q = _llm(system, prompt, max_tokens=400).strip()
    lines = [l for l in q.splitlines() if not l.lower().startswith("format") and "?" not in l or l.endswith("?")]
    return "\n".join(lines).strip() or q


def _score(cfg: InterviewConfig, history: list[InterviewTurn]) -> ScoreCard:
    transcript = "\n".join(
        f"Q{i+1}: {t.question}\nA{i+1}: {t.answer}" for i, t in enumerate(history)
    )
    prompt = (
        f"Course: {cfg.course} | Candidate: {cfg.candidate_name} | "
        f"Level: {cfg.skill_level} | Type: {cfg.interview_type}\n\n"
        f"Full interview transcript:\n{transcript}"
    )
    raw = _llm(_SCORER_SYSTEM, prompt, json_mode=True, max_tokens=2000)
    data = _parse_json(raw) or {}

    costs = {}
    for mins in [10, 20, 30]:
        tokens = mins * 200
        cost_usd = tokens * 0.00000035
        costs[str(mins)] = f"~${cost_usd:.3f}"

    return ScoreCard(
        overall_score=int(data.get("overall_score", 50)),
        technical_knowledge=int(data.get("technical_knowledge", 50)),
        communication=int(data.get("communication", 50)),
        problem_solving=int(data.get("problem_solving", 50)),
        answer_structure=int(data.get("answer_structure", 50)),
        strengths=list(data.get("strengths", ["Shows effort"])),
        improvements=list(data.get("improvements", ["Needs more depth"])),
        recommendations=list(data.get("recommendations", ["Practice more problems"])),
        cost_estimate=costs,
        summary=str(data.get("summary", "Interview completed.")),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/start", response_model=StartOut)
def start_interview(cfg: InterviewConfig):
    session_id = str(uuid.uuid4())
    first_q = _opening_question(cfg)
    return StartOut(session_id=session_id, first_question=first_q)


@router.post("/respond", response_model=RespondOut)
def respond(body: RespondIn):
    if body.questions_asked >= body.config.num_questions:
        return RespondOut(done=True)
    next_q = _follow_up(body.config, body.history, body.latest_answer)
    return RespondOut(next_question=next_q, done=False)


@router.post("/finish", response_model=FinishOut)
def finish_interview(body: FinishIn):
    scorecard = _score(body.config, body.history)
    return FinishOut(scorecard=scorecard)