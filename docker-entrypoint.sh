#!/usr/bin/env bash
set -e

# Build the knowledge base on first run if it hasn't been built yet.
if [ ! -d "athena/data/store/chroma" ] || [ -z "$(ls -A athena/data/store/chroma 2>/dev/null)" ]; then
  if [ -n "$GEMINI_API_KEY" ]; then
    echo "[entrypoint] Building knowledge base (first run)…"
    python -m athena.scripts.build_index || echo "[entrypoint] build failed; the app will show a warning."
  else
    echo "[entrypoint] GEMINI_API_KEY not set; skipping index build."
  fi
fi

exec streamlit run athena/ui/app.py
