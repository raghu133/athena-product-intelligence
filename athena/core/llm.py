"""Gemini client wrapper — the only module that talks to the Gemini API.

Responsibilities:
  * lazy singleton client (so importing this module never requires a key)
  * text generation (fast / deep / bulk model selection)
  * strict-JSON generation with schema hinting + robust parsing
  * batched embeddings for RAG
  * retry with exponential backoff on transient errors
  * optional trace-span recording for observability

Centralizing all API access here means retries, model routing, and tracing are
implemented once and used everywhere.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from athena.core.config import settings
from athena.core.tracing import Trace

_client = None


class LLMNotConfigured(RuntimeError):
    """Raised when an API call is attempted without a configured key."""


def _get_client():
    """Lazily create the google-genai client. Kept lazy so unit tests and
    dataset generation that don't need the network can import freely."""
    global _client
    if _client is None:
        if not settings.has_api_key:
            raise LLMNotConfigured(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
                "(free at https://aistudio.google.com/apikey)."
            )
        from google import genai  # imported lazily
        _client = genai.Client(api_key=settings.api_key)
    return _client


class _Transient(Exception):
    """Wraps a retryable API error. Carries the server-advised retry delay (s)
    parsed from a 429 RESOURCE_EXHAUSTED response, so we can wait exactly as long
    as the free-tier quota window requires instead of guessing."""
    def __init__(self, message: str, retry_after: float = 0.0):
        super().__init__(message)
        self.retry_after = retry_after


class QuotaExhausted(RuntimeError):
    """The daily free-tier quota is used up. Retrying is futile (resets at
    midnight US Pacific), so we fail fast with a clear, actionable message
    instead of burning attempts."""


def _is_daily_quota(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "perday" in msg or "requests_per_day" in msg or "requestsperday" in msg


def _is_transient(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in ("429", "resource_exhausted", "unavailable", "503",
                                  "500", "deadline", "timeout", "rate"))


def _parse_retry_after(msg: str) -> float:
    """Extract the server's advised wait from a 429 ('Please retry in 28.4s' or
    'retryDelay': '28s'). Returns 0 if none found."""
    m = re.search(r"retry(?:delay)?['\"]?\s*[:in]*\s*['\"]?(\d+(?:\.\d+)?)s", msg, re.I)
    return float(m.group(1)) if m else 0.0


# Free-tier quota can force waits of ~30s; allow enough attempts/time to ride
# through a per-minute rate-limit window rather than failing the whole build.
def _wait_strategy(retry_state) -> float:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    advised = getattr(exc, "retry_after", 0.0) if exc else 0.0
    backoff = min(2.0 * (2 ** (retry_state.attempt_number - 1)), 30.0)
    return max(advised + 1.0, backoff)  # honor server delay, +1s safety margin


_retry = retry(
    stop=stop_after_attempt(6),
    wait=_wait_strategy,
    retry=retry_if_exception_type(_Transient),
    reraise=True,
)


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        if _is_daily_quota(exc):
            raise QuotaExhausted(
                "Gemini daily free-tier quota is exhausted (resets ~midnight US "
                "Pacific). Any embeddings computed so far are cached, so re-running "
                "the build later resumes where it stopped."
            ) from exc
        if _is_transient(exc):
            raise _Transient(str(exc), _parse_retry_after(str(exc))) from exc
        raise


# --- Text generation -----------------------------------------------------
@_retry
def generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    thinking: bool = True,
    trace: Optional[Trace] = None,
    span_name: str = "generate",
) -> str:
    """Generate text. `model` defaults to the fast model.

    Gemini 2.5 models are 'thinking' models that spend output tokens on internal
    reasoning before answering. For deep synthesis we keep thinking on; for short
    utility outputs set thinking=False so the token budget isn't consumed by
    reasoning (which can otherwise return an empty answer)."""
    from google.genai import types

    client = _get_client()
    model = model or settings.model_fast
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system,
        thinking_config=None if thinking else types.ThinkingConfig(thinking_budget=0),
    )

    def _do() -> str:
        resp = _call(
            client.models.generate_content,
            model=model,
            contents=prompt,
            config=cfg,
        )
        return (resp.text or "").strip()

    if trace is not None:
        with trace.span(span_name, "llm", model=model, prompt=prompt) as sp:
            out = _do()
            sp.outputs = {"text": out}
            sp.metadata["tokens"] = _approx_tokens(prompt) + _approx_tokens(out)
            return out
    return _do()


@_retry
def generate_json(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    thinking: bool = False,
    trace: Optional[Trace] = None,
    span_name: str = "generate_json",
) -> Any:
    """Generate and parse strict JSON. Uses Gemini's JSON response mode and a
    defensive fallback parser for the occasional stray markdown fence.

    Thinking defaults OFF here: structured extraction/planning/rerank calls are
    short and benefit from the full token budget going to the JSON output."""
    from google.genai import types

    client = _get_client()
    model = model or settings.model_fast
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system,
        response_mime_type="application/json",
        thinking_config=None if thinking else types.ThinkingConfig(thinking_budget=0),
    )

    def _do() -> Any:
        resp = _call(
            client.models.generate_content,
            model=model,
            contents=prompt,
            config=cfg,
        )
        return _parse_json(resp.text or "")

    if trace is not None:
        with trace.span(span_name, "llm", model=model, prompt=prompt) as sp:
            out = _do()
            sp.outputs = {"json": out}
            sp.metadata["tokens"] = _approx_tokens(prompt)
            return out
    return _do()


# --- Embeddings ----------------------------------------------------------
@_retry
def embed(
    texts: list[str],
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
    trace: Optional[Trace] = None,
) -> list[list[float]]:
    """Embed a batch of texts. `task_type` is RETRIEVAL_DOCUMENT for indexing
    and RETRIEVAL_QUERY for search — asymmetric embeddings improve recall."""
    from google.genai import types

    client = _get_client()
    vectors: list[list[float]] = []
    batch = settings.embed_batch_size

    def _do_batch(chunk: list[str]) -> list[list[float]]:
        resp = _call(
            client.models.embed_content,
            model=settings.embed_model,
            contents=chunk,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=settings.embed_dim,
            ),
        )
        # Non-3072 output dims must be L2-normalized for correct cosine similarity
        # (per Gemini embedding docs). We store 768-dim, so normalize here.
        return [_l2_normalize(list(e.values)) for e in resp.embeddings]

    def _run() -> None:
        for i in range(0, len(texts), batch):
            vectors.extend(_do_batch(texts[i:i + batch]))
            # Gentle pacing to stay under free-tier rate limits on large corpora.
            if i + batch < len(texts):
                time.sleep(0.2)

    if trace is not None:
        with trace.span("embed", "retrieval", n=len(texts), task_type=task_type) as sp:
            _run()
            sp.outputs = {"n_vectors": len(vectors)}
    else:
        _run()
    return vectors


def embed_query(text: str, trace: Optional[Trace] = None) -> list[float]:
    return embed([text], task_type="RETRIEVAL_QUERY", trace=trace)[0]


# --- helpers -------------------------------------------------------------
def _l2_normalize(vec: list[float]) -> list[float]:
    norm = sum(v * v for v in vec) ** 0.5
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _parse_json(raw: str) -> Any:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # strip ```json fences if present
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass
    # last resort: grab the outermost {...} or [...]
    for opener, closer in (("{", "}"), ("[", "]")):
        s, e = raw.find(opener), raw.rfind(closer)
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(raw[s:e + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from model output: {raw[:300]}")
