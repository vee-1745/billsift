# Public demo image (works on Render, Hugging Face Spaces, Fly.io, any Docker host).
# Runs in BILLSIFT_DEMO mode: rule-based extraction only, per-visitor in-memory data.
FROM python:3.12-slim

# OpenCV (pulled in by RapidOCR) needs these shared libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Demo mode, sample bills only (uploads would run OCR, which needs more than a 512 MB free instance)
ENV BILLSIFT_DEMO=1 \
    BILLSIFT_DEMO_UPLOADS=0 \
    PYTHONUNBUFFERED=1

# Render sets $PORT; default 8501 elsewhere. 5 MB upload cap keeps the demo small.
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --server.maxUploadSize=5 --browser.gatherUsageStats=false"]
