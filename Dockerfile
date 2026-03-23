FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy only pyproject.toml first — installs deps as a cached layer
COPY pyproject.toml .
# Intentional Docker layer-caching pattern: pip install -e . creates a .egg-link
# pointing to /app. When COPY . . runs, the code becomes available automatically.
RUN pip install --no-cache-dir -e .

# Download NLTK punkt tokenizer at build time (avoids runtime download)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# PYTHONPATH ensures packages/ sub-packages are importable at runtime
ENV PYTHONPATH=/app/packages:/app

COPY . .

CMD ["python", "-m", "packages.bot.main"]
