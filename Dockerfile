FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && mkdir -p /private_data \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt ./requirements-docker.txt
COPY requirements-dashboard.txt ./requirements-dashboard.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-docker.txt \
    && python -m pip install -r requirements-dashboard.txt

COPY api ./api
COPY src ./src
COPY models ./models
COPY dashboard ./dashboard
COPY dashboard_data ./dashboard_data

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
