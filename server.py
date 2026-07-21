"""
Flask server for the redesigned, multilingual Emotion Detector.

This is a complete rewrite of the original server. The old version
rejected common conversational phrases like "i want my ex back" or
"μου λείπει ο/η πρώην μου" with a generic "Invalid text!" error. The new
version normalizes the input, detects the language, and returns a rich
JSON payload that the dashboard UI renders as a sentiment arc, an emotion
distribution bar chart, a confidence graph, and a human summary.
"""

from __future__ import annotations

import json
import os

import requests
from flask import Flask, jsonify, render_template, request

from EmotionDetection.emotion_detection import emotion_detector
from EmotionDetection.wellbeing import crisis_resources_for, detect_crisis

app = Flask("Emotion Detector")

# ---------------------------------------------------------------------------
# AI wellbeing companion chat (uses the Anthropic Messages API directly).
#
# Set your own key before running:
#   export ANTHROPIC_API_KEY="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"
# Get one at https://console.anthropic.com/settings/keys
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-5"

COMPANION_SYSTEM_PROMPT = """You are a warm, supportive wellbeing companion \
built into a multilingual emotion-detection app called "Emotion Detector".

Hard rules, never break these:
- You are NOT a licensed psychologist, therapist, or doctor. Never claim to \
be one, never diagnose a condition, never prescribe medication or treatment.
- ALWAYS reply in the same language the user writes in (use the provided \
detected language as your default/starting point).
- Keep replies conversational and human: a few warm sentences, not a \
clinical report, and no bullet lists during the conversation itself (a \
bullet-point summary is generated separately at the end of the session).
- If the user's message is positive (joy, pride, excitement, good news): \
genuinely celebrate with them, be specific about what's great, then gently \
ask if they'd like to keep talking about that or about anything else on \
their mind. Do not manufacture a problem to solve.
- If the user's message is negative or difficult (sadness, anger, fear, \
grief, stress, etc.): lead with empathy, validate the feeling without \
judgment, ask one gentle open-ended question to understand more, and where \
appropriate offer one or two concrete, practical, evidence-informed coping \
ideas (grounding techniques, journaling, reaching out to someone they \
trust, brief movement or breathing, sleep/routine basics) — not empty \
platitudes.
- Never attach a diagnostic label to the user, even casually.
- If the user expresses thoughts of self-harm, suicide, or harming someone \
else, respond with care and take it seriously; the app automatically \
surfaces verified local crisis resources alongside your reply, so you \
don't need to invent phone numbers yourself — just acknowledge it warmly \
and encourage them to use those resources or talk to someone right now.
- Naturally, and without being repetitive about it, make clear you're an \
AI companion, not a replacement for a licensed mental health professional \
if something feels like it needs real support.
"""

SUMMARY_SYSTEM_PROMPT = """The wellbeing chat session below is ending. \
Based ONLY on what was actually discussed, write a short bullet-point list \
(4-7 bullets) in the same language as the conversation, of concrete, \
gentle, actionable things the user could try to feel psychologically \
better or to keep doing well. Ground every bullet in something that was \
actually said — don't invent generic advice unrelated to the conversation. \
If the conversation was positive throughout, the bullets can be about \
sustaining and celebrating what's going well rather than "fixing" \
anything. Never diagnose. Never mention medication. Output ONLY the \
bullet points, one per line, each starting with "- ", and nothing else \
(no heading, no preamble, no closing remark).
"""


def _call_anthropic(system_prompt: str, messages: list, max_tokens: int = 500) -> str:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set on the server. "
            "Set it as an environment variable before starting server.py."
        )
    resp = requests.post(
        ANTHROPIC_URL,
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    return "\n".join(parts).strip()


# Permissive CORS so the dashboard works whether it's served by Flask itself
# (same-origin, port 5000) or opened as a static file from a different origin
# (e.g. a live-preview plugin, file://, or a separate dev server). Without
# these headers, browsers block the /analyze POST with a CORS error and the
# frontend shows a generic "Connection error" toast.
_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "3600",
}


@app.after_request
def _add_cors(response):
    for k, v in _CORS_HEADERS.items():
        response.headers.setdefault(k, v)
    return response


@app.route("/")
def render_index_page():
    """Renders the redesigned dashboard UI."""
    return render_template("index.html")


@app.route("/analyze", methods=["OPTIONS"])
def _analyze_options():
    """CORS preflight for the /analyze endpoint."""
    return ("", 204, _CORS_HEADERS)


@app.route("/analyze", methods=["POST"])
def analyze():
    """JSON endpoint used by the dashboard UI.

    Body: {"text": "..."}  (also accepts ?textToAnalyse=... on the query string
    for backwards compatibility with the original Watson-NLP lab endpoint).
    Returns: the full emotion_detector() dict.
    """
    payload = request.get_json(silent=True) or {}
    text = (
        payload.get("text", "")
        or request.args.get("textToAnalyse", "")
        or request.args.get("text", "")
    )
    result = emotion_detector(text)
    return jsonify(result), (200 if result.get("ok") else 400)


# Backwards-compat with the original `/emotionDetector?textToAnalyse=...`
# route so older clients / tests still work. Returns the rich JSON.
@app.route("/emotionDetector")
def sent_detector():
    text = request.args.get("textToAnalyse", "")
    result = emotion_detector(text)
    status = 200 if result.get("ok") else 400
    response = app.response_class(
        response=json.dumps(result, ensure_ascii=False),
        status=status,
        mimetype="application/json",
    )
    return response


@app.route("/chat", methods=["OPTIONS"])
def _chat_options():
    return ("", 204, _CORS_HEADERS)


@app.route("/chat", methods=["POST"])
def chat():
    """One turn of the wellbeing companion chat.

    Body: {
      "message": "...",              # the user's new message
      "history": [{"role": "user"|"assistant", "content": "..."}, ...],
      "language": "Greek",
      "language_code": "el",
      "primary_emotion": "sadness"
    }
    Returns: {"ok": true, "reply": "...", "crisis": bool,
              "crisis_resources": {...} | null}
    """
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    history = payload.get("history") or []
    lang_name = payload.get("language") or "English"
    lang_code = payload.get("language_code") or "en"
    primary_emotion = payload.get("primary_emotion") or ""

    if not message:
        return jsonify({"ok": False, "message": "Empty message."}), 400

    is_crisis = detect_crisis(message, lang_code)

    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    context_note = (
        f"[detected_language: {lang_name} ({lang_code}); "
        f"primary_emotion_from_analysis: {primary_emotion or 'n/a'}]\n{message}"
    )
    messages.append({"role": "user", "content": context_note})

    try:
        reply = _call_anthropic(COMPANION_SYSTEM_PROMPT, messages)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "message": f"Chat service error: {exc}"}), 502

    return jsonify({
        "ok": True,
        "reply": reply,
        "crisis": is_crisis,
        "crisis_resources": crisis_resources_for(lang_code) if is_crisis else None,
    })


@app.route("/chat/end", methods=["OPTIONS"])
def _chat_end_options():
    return ("", 204, _CORS_HEADERS)


@app.route("/chat/end", methods=["POST"])
def chat_end():
    """Produces the end-of-session bullet-point summary.

    Body: {"history": [...], "language": "Greek"}
    Returns: {"ok": true, "bullets": ["...", "...", ...]}
    """
    payload = request.get_json(silent=True) or {}
    history = payload.get("history") or []
    lang_name = payload.get("language") or "English"

    if not history:
        return jsonify({"ok": False, "message": "No conversation to summarize."}), 400

    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    messages.append({
        "role": "user",
        "content": f"[session ending; respond in {lang_name}] Please summarize now.",
    })

    try:
        raw = _call_anthropic(SUMMARY_SYSTEM_PROMPT, messages, max_tokens=400)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "message": f"Chat service error: {exc}"}), 502

    bullets = [
        line.lstrip("-•* ").strip()
        for line in raw.splitlines()
        if line.strip().lstrip("-•* ").strip()
    ]
    return jsonify({"ok": True, "bullets": bullets})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)