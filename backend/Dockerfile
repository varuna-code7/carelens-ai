FROM python:3.11-slim

# Tesseract is a system-level program, not a Python package —
# pytesseract just calls out to it, so it must be installed here.
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets $PORT at runtime; gunicorn binds to it.
CMD gunicorn --bind 0.0.0.0:$PORT app:app