# Deployment Guide

Athena runs locally and containerizes cleanly for any host. All state (indexes,
memory, traces) is local files, so persistence is a single volume.

---

## Prerequisites
- A free **Gemini API key**: https://aistudio.google.com/apikey
- Python 3.11+ (local) **or** Docker (containers)

---

## Option A — Local (fastest to demo)

```bash
pip install -r requirements.txt
cp .env.example .env          # add GEMINI_API_KEY
python -m athena.scripts.build_index
streamlit run athena/ui/app.py
```
Open http://localhost:8501.

---

## Option B — Docker (deploy-ready)

```bash
cp .env.example .env          # add GEMINI_API_KEY
docker compose up --build
```
The entrypoint builds the knowledge base on first run and persists it in the
`athena_store` volume. Open http://localhost:8501.

---

## Option C — Hosted (free live URL)

### Streamlit Community Cloud
1. Push this repo to GitHub.
2. On https://share.streamlit.io, create an app pointing at
   `athena/ui/app.py`.
3. In the app's **Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
4. The app boots; on first load it will prompt to build the index. For hosted
   builds, either commit a prebuilt `athena/data/store/` or add a small startup
   hook that runs `build_index` once. (Locally, just run the build script.)

### Render / Railway / Fly.io (container)
- Point the platform at the `Dockerfile`.
- Set env var `GEMINI_API_KEY`.
- Expose port `8501`.
- Attach a persistent disk mounted at `/app/athena/data/store` so the index and
  memory survive restarts.

---

## Configuration

All knobs live in `athena/core/config.py`; override via env vars (see
`.env.example`). Notable ones:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | required |
| `GEMINI_MODEL_FAST` | `gemini-2.5-flash` | agents, rerank, RAG |
| `GEMINI_MODEL_DEEP` | `gemini-2.5-pro` | synthesis, reports |
| `GEMINI_MODEL_BULK` | `gemini-2.5-flash-lite` | bulk enrichment |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-001` | embeddings |

---

## Rebuilding / resetting

```bash
python -m athena.scripts.build_index --force-regen   # regenerate dataset + reindex
```
Delete `athena/data/store/` to wipe indexes, memory, and traces.

---

## Health checks

```bash
python -m pytest tests/ -q                 # offline unit tests (no key needed)
python -m athena.scripts.run_eval          # end-to-end eval (needs key + index)
```
