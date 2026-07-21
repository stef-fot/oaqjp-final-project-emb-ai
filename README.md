<div align="center">

# 🎭 Emotion Detector

**Multilingual AI-powered emotion & sentiment analysis.**

Type or paste any text in any of **1000+ languages** and the app detects the language, scores 5 core emotions (joy, sadness, anger, fear, disgust) + surprise, returns a sentiment arc, a confidence score, and a human summary — all in under 100&nbsp;ms.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3-000000?logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-Apache_2.0-blue)
![No external API required](https://img.shields.io/badge/Runs_offline-100%25_on--device-27e0a4)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Running Locally](#running-locally)
- [Using the Web UI](#using-the-web-ui)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [License & Credits](#license--credits)

---

## Overview

**Emotion Detector** is a small Flask web app that analyzes free-form text and returns a rich emotion + sentiment profile. It is the evolution of an original Watson-NLP lab project — the rewrite drops the dependency on a remote NLP service in favor of a **precompiled, in-process multilingual lexicon** that runs entirely offline.

The old version rejected common conversational phrases like *"i want my ex back"* or *"μου λείπει ο/η πρώην μου"* with a generic *"Invalid text!"* error. The new version:

- Normalizes the input (strips control chars, URLs, repeated-gibberish runs).
- Detects the script → maps to a language (Unicode block-based, zero dependencies).
- Scores 5 core emotions + surprise against a hand-curated multilingual lexicon covering 50+ of the world's most spoken languages.
- Optionally blends in **Watson NLP** (English-only) as a soft upstream when reachable and under 250&nbsp;ms.
- Returns a rich JSON payload rendered by a dashboard UI as a sentiment gauge, an emotion distribution bar chart, a confidence graph, and a human-readable summary.

---

## Key Features

| Feature | Description |
| --- | --- |
| 🌍 **1000+ languages** | Unicode-script detection + a 50+ language hand-curated lexicon; graceful fallback for unknown scripts. |
| ⚡ **Sub-100&nbsp;ms latency** | Lexicon lookups are O(n) and fully in-memory. No network round-trip on the hot path. |
| 📊 **Rich dashboard** | Sentiment arc, 5+1 emotion bars, confidence graph, multilingual greeting, language flag, elapsed-time pill. |
| 🎯 **5 core emotions + surprise** | Joy, sadness, anger, fear, disgust, surprise — each scored 0–100%. |
| 💬 **Sentiment** | Polarity score (-1..+1), positivity (0..100%), label (`positive` / `neutral` / `negative`). |
| 🔁 **Optional Watson NLP** | If enabled and reachable, English text is blended 50/50 with Watson's output. |
| 🌐 **CORS enabled** | The `/analyze` endpoint is callable from any origin (useful for live-preview and embedded widgets). |
| 🛡️ **Defensive input handling** | Strips control chars, URLs, repeated-gibberish runs; caps input at 4&nbsp;000 chars; rejects pure-punctuation input politely. |
| ☁️ **Cloud-ready** | Ships with a Cloud Function / Cloud Run entrypoint (`cloud_function/main.py`) and an `app.yaml`. |

---

## How It Works

```
┌────────────────────┐  POST /analyze   ┌─────────────────┐  in-process   ┌────────────────┐
│  Browser (UI)      │  { "text": "…" } │  Flask server   │  lexicon      │  emotion_      │
│  http://:5000/     │ ───────────────► │  /analyze       │ ────────────► │  detector()    │
│  templates/        │ ◄─────────────── │  port 5000      │ ◄──────────── │  + optional    │
│  index.html        │   rich JSON      │                 │   5+1 emo +   │  Watson NLP    │
└────────────────────┘                  └─────────────────┘   sentiment    └────────────────┘
```

1. **Input is normalized** — control chars, URLs, and repeated-gibberish runs are stripped; length is capped at 4 000 chars.
2. **Language is detected** from the dominant Unicode script (Latin, Cyrillic, Han, Devanagari, Arabic, …) with a Latin-script disambiguation pass against the lexicon itself.
3. **Emotion counts** are accumulated by scanning the text against per-language keyword lists. Each match adds 1.0; English fallback matches add 0.5.
4. **A Dirichlet-like prior** (0.5 per class) is added to the counts and the result is normalized to a distribution.
5. **Sentiment** is computed as a weighted sum: `joy +1.00, surprise +0.10, neutral 0, fear −0.30, sadness −0.70, anger −0.85, disgust −0.80`, then mapped to a 0–100% positivity.
6. **Confidence** blends a length factor (`min(1, len/200)`) and the top-1-vs-top-2 margin in the distribution.
7. **Optional Watson NLP** (English only, 250 ms timeout) is blended 50/50 when reachable.

The whole pipeline is **<1 ms** for typical inputs.

---

## Project Structure

```
Emotions-Detection-main/
├── server.py                         # Flask app: /, /analyze, /emotionDetector
├── requirements.txt                  # flask, requests, gunicorn
├── app.yaml                          # Google App Engine (Flex / Cloud Run) config
├── _smoke.py                         # Quick CLI smoke test
├── _where.py                         # Verify the EmotionDetection package is importable
├── __init__.py                       # Top-level package marker
│
├── EmotionDetection/
│   ├── __init__.py                   # Re-exports emotion_detector
│   └── emotion_detection.py          # Multilingual lexicon + analyzer
│
├── templates/
│   └── index.html                    # Dashboard UI (HTML + CSS + vanilla JS)
│
├── cloud_function/
│   ├── main.py                       # GCP Cloud Function / Cloud Run entrypoint
│   │                                 #   + AWS Lambda adapter
│   └── README.md                     # Deploy instructions
│
└── README.md                         # ← you are here
```

---

## Tech Stack

- **Python 3.10+** (3.11+ recommended; tested on 3.11 / 3.14)
- **Flask 3** — HTTP server + Jinja2 template rendering
- **Werkzeug** — WSGI (Flask's built-in dev server)
- **Gunicorn** — production WSGI server (used by `app.yaml`)
- **requests** — optional, only if you enable the Watson NLP upstream
- **Zero ML dependencies** — the analyzer is a curated keyword/lexicon model that runs entirely in-process
- **Vanilla HTML + CSS + JS** frontend (no build step)

---

## Quick Start

```powershell
# 1. Clone & enter
cd C:\path\to\Emotions-Detection-main

# 2. Install
pip install -r requirements.txt

# 3. Run
python server.py
```

Open **http://localhost:5000** in your browser. Type or paste any text in any language and click **✨ Analyze Emotion**. You should see a sentiment gauge, an emotion bar chart, and a confidence graph appear on the right.

That's it. No API keys, no databases, no extra services to start.

---

## Running Locally

### Option A — Flask dev server (simplest)

```bash
python server.py
```

Listens on `0.0.0.0:5000` (reachable from `127.0.0.1`, `localhost`, and your LAN IP). Debug mode is off. Auto-reload is off; restart the process to pick up code changes.

### Option B — Gunicorn (production-like)

```bash
gunicorn -b 127.0.0.1:5000 -w 2 --timeout 30 server:app
```

`-w 2` = 2 worker processes; bump it for higher throughput. The same command is what `app.yaml` uses on App Engine / Cloud Run.

### Smoke test (CLI, no browser)

```bash
python _smoke.py
```

Runs a handful of samples (English, Greek, Spanish, Japanese, gibberish) through the analyzer and prints a one-line summary for each. Useful after any change to `emotion_detection.py`.

### Verify the package is importable

```bash
python _where.py
```

Prints the location of the `EmotionDetection` package as Python sees it — useful when debugging "ModuleNotFoundError" after moving files around.

---

## Using the Web UI

1. Open **http://localhost:5000**.
2. Type or paste any text. Examples (clickable chips):
   - `i want my ex back`
   - `μου λείπει ο/η πρώην μου`
   - `I just got promoted today!`
   - `Estoy muy enojado contigo`
   - `今日とても嬉しいです`
   - `This is the worst day ever`
3. Hit **✨ Analyze Emotion** (or `Ctrl + Enter`).
4. The right panel shows:
   - **Detected language** + a "Hello!" greeting in that language.
   - **Sentiment arc** — a 0–100% positivity gauge with Negative / Neutral / Positive labels.
   - **Emotion distribution** — bars for joy, sadness, anger, fear, disgust, surprise.
   - **Confidence graph** — animated line climbing to the final confidence score.
   - **Meta footer** — word count, model version, language, elapsed time.

You can also call the API directly from `curl` (see below).

---

## API Reference

### `POST /analyze`

Primary endpoint used by the dashboard UI.

**Request**

```http
POST /analyze HTTP/1.1
Host: localhost:5000
Content-Type: application/json

{ "text": "i want my ex back" }
```

The endpoint also accepts the legacy `?textToAnalyse=...` (and plain `?text=...`) query string on the same route for backwards compatibility.

**Response — `200 OK`** (success)

```json
{
  "ok": true,
  "message": null,
  "language": "English",
  "language_code": "en",
  "greeting": "Hello!",
  "primary_emotion": "sadness",
  "distribution": { "joy": 0.13, "sadness": 0.36, "anger": 0.16, "fear": 0.16, "disgust": 0.15, "surprise": 0.04 },
  "distribution_percent": { "joy": 13.0, "sadness": 36.0, "anger": 16.0, "fear": 16.0, "disgust": 15.0, "surprise": 4.0 },
  "sentiment": { "score": -0.07, "positivity": 46.5, "label": "neutral" },
  "confidence": 0.75,
  "summary": "Detected primary emotions of sadness and anger in English.",
  "elapsed_ms": 1.2,
  "normalized_text": "i want my ex back",
  "word_count": 5
}
```

**Response — `400 Bad Request`** (input rejected, e.g. empty / pure-punctuation)

```json
{
  "ok": false,
  "message": "Please enter some text to analyze — even a single sentence is fine.",
  "elapsed_ms": 0.0
}
```

**CORS**

`Access-Control-Allow-Origin: *` is set on every response, and `OPTIONS /analyze` is handled for preflight, so the endpoint is safe to call from any browser origin.

### `GET /emotionDetector?textToAnalyse=...`

Backwards-compatible wrapper that returns the **same JSON** as `/analyze` but as a `GET` with the text on the query string. Kept so older clients and the original lab's tests still work.

```bash
curl "http://localhost:5000/emotionDetector?textToAnalyse=i%20just%20got%20promoted"
```

### `GET /`

Renders `templates/index.html` — the dashboard UI.

### Quick test with curl

```bash
curl -X POST http://localhost:5000/analyze \
     -H "Content-Type: application/json" \
     -d '{"text":"i want my ex back"}'
```

Or PowerShell:

```powershell
Invoke-RestMethod -Method POST -Uri http://localhost:5000/analyze `
     -ContentType 'application/json' `
     -Body '{"text":"i want my ex back"}'
```

---

## Deployment

### Google Cloud Run (recommended)

```bash
gcloud run deploy emotion-detector \
  --source . --region us-central1 \
  --allow-unauthenticated --memory 256Mi --cpu 1
```

`app.yaml` is preconfigured for **App Engine Flex** with the same command:

```yaml
entrypoint: gunicorn -b :$PORT server:app --workers 2 --timeout 30
```

### Google Cloud Functions

```bash
gcloud functions deploy emotion-detector \
  --runtime python311 \
  --trigger-http \
  --allow-unauthenticated \
  --memory 256MB \
  --timeout 10s \
  --entry-point analyze \
  --source ./cloud_function
```

See `cloud_function/README.md` for the full Cloud Function / Cloud Run / AWS Lambda adapter details. The function also caches results in-process (per warm instance) using an LRU keyed on a SHA-1 of the normalized text, so repeated phrases (very common in conversational UIs) skip almost all work after the first call.

### Local Docker (optional)

A minimal `Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "2", "--timeout", "30", "server:app"]
```

Build & run:

```bash
docker build -t emotion-detector .
docker run --rm -p 5000:5000 emotion-detector
```

---

## Troubleshooting

### 🔌 "Couldn't reach the server" / "Connection error" toast

This toast now includes the URL the frontend actually tried and the underlying network error. Common causes:

1. **Backend isn't running.**
   Start it (`python server.py`) and confirm the terminal shows
   `Running on http://127.0.0.1:5000`. Hit it directly:
   ```bash
   curl http://127.0.0.1:5000/analyze -X POST \
        -H "Content-Type: application/json" -d '{"text":"hello"}'
   ```
   You should get JSON back.

2. **The page is opened from `file://` (double-click on `index.html`).**
   `fetch("/analyze", …)` resolves to `file:///analyze`, which the OS can't serve.
   The frontend detects this and falls back to `http://127.0.0.1:5000/analyze`; the toast will also include a one-line hint. **Always open the page through Flask** — go to `http://localhost:5000` in your browser, not by double-clicking the file.

3. **CORS preflight blocked.**
   The server now sends `Access-Control-Allow-Origin: *` on every response and answers `OPTIONS /analyze` with 204. If you still see a CORS error in DevTools, check that no corporate proxy / browser extension is stripping response headers.

4. **Mixed content (HTTPS page → HTTP backend).**
   Browsers block `https://` pages from calling `http://` endpoints. Either serve Flask over HTTPS (terminate TLS in front of Gunicorn, or use Cloud Run), or open the dashboard at `http://localhost:5000` while developing.

5. **Wrong host / port.**
   If you started Flask on a non-default port, the dashboard's relative `/analyze` will still reach it (it's same-origin) — but if you opened the page from a different origin (e.g. `127.0.0.1:3000` via a live-preview plugin), the browser will use *that* origin. The toast now shows the exact URL it tried, so the mismatch is obvious in the error message.

### 📦 `ModuleNotFoundError: No module named 'EmotionDetection'`

The analyzer is imported as a package, so the project root must be on `sys.path`. Either:

- Run from the project root: `cd Emotions-Detection-main && python server.py`.
- Or install the package: `pip install -e .` (add a `pyproject.toml` first if you don't have one).

`python _where.py` will print the resolved package path — if it says `None`, the package isn't discoverable.

### 🐢 Slow first request

Cold-start on the dev server is essentially zero (the lexicon is just a dict). If you see latency >100&nbsp;ms, the optional **Watson NLP** upstream is being consulted and timing out (250&nbsp;ms budget). This is expected on a cold instance. Disable Watson (it's off by default: `WATSON_UPSTREAM_ENABLED = False`) if you don't need it.

### 🐛 Browser console shows "CORS policy: No 'Access-Control-Allow-Origin' header…"

Should not happen on the current build — both the `@app.after_request` hook and the explicit `OPTIONS /analyze` handler emit the headers. If you still see it, hard-reload (Ctrl+Shift+R) to bypass cached HTML, and confirm your reverse proxy isn't stripping the headers.

### 📝 Adding new keywords / new languages

The lexicon is a plain dict at the top of `EmotionDetection/emotion_detection.py`. To add words for a new language, append them under the appropriate ISO-639-1 code in the per-emotion lists. To add a new language to the greetings table, add it to `GREETINGS`. Restart the server — no other code change is required.

### 🧹 Resetting for a fresh start

The app is fully stateless (no database, no on-disk cache beyond the Cloud Function's optional LRU). Just kill the process and restart.

---

## License & Credits

- **License:** Apache 2.0 (inherited from the original Watson-NLP lab starter).
- **Lexicon:** hand-curated keyword lists for 50+ languages — see `EmotionDetection/emotion_detection.py`.
- **Original IBM Watson NLP library:** used in the legacy `emotion_detection` path; the rewrite keeps it as a *soft, optional* upstream, never a hard dependency.

<div align="center">

Happy analyzing! 🎭

</div>
