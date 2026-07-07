# Project North Star — Streamlit app container.
#
# Secrets (GEMINI_API_KEY, TAVILY_API_KEY, NEON_CONNECTION_STRING) are
# never baked into this image — inject them as real environment variables
# at container-run time (e.g. `docker run -e GEMINI_API_KEY=... `, or the
# platform's own secrets mechanism for HF Spaces / similar). See
# .env.example for the full list.

FROM python:3.12-slim

WORKDIR /app

# Dependency layer first (requirements.txt), so it's cached independently
# of source-code changes — requirements.txt mirrors pyproject.toml's
# [project.dependencies]; see that file for the real dependency source.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source layout: src/ (package-dir root, per pyproject.toml's
# [tool.setuptools] config), .agent/skills/ (the Verification Question
# Generator Skill — required at runtime, not just dev-time, since
# agents/coaching_pace_agent.py adds it to sys.path relative to this
# WORKDIR), streamlit_app.py (the actual `streamlit run` target), and
# pyproject.toml itself (needed for the local package install below).
COPY src/ ./src/
COPY .agent/ ./.agent/
COPY streamlit_app.py .
COPY pyproject.toml .

# Register the local package in editable mode (src/ -> importable `main`,
# `agents`, `data`, etc.) without re-resolving dependencies already
# installed above. Editable, not a real install, is deliberate here, not
# a dev-only shortcut: `agents/coaching_pace_agent.py` resolves the
# Verification Skill's `.agent/skills/` path relative to its own
# `__file__` at runtime — a non-editable install copies source into
# site-packages, breaking that path resolution structurally (verified: a
# real `pip install --no-deps .` reproduces exactly the "No module named
# main" / "No module named verification_question_generator" failures a
# non-editable install causes here). Editable install keeps every
# module's `__file__` pointing at this image's real `/app/src/...`
# location, matching local dev exactly.
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
