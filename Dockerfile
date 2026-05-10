# Multi-purpose container for the InsightForge Next.js frontend and FastAPI backend.
FROM node:20-bookworm AS frontend
WORKDIR /app
COPY package.json ./
RUN npm install
COPY app ./app
COPY next.config.mjs tsconfig.json eslint.config.mjs next-env.d.ts ./
RUN npm run build

FROM python:3.11-slim AS backend
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MAX_MEMORY_ROWS=250000
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY --from=frontend /app/.next ./frontend/.next
COPY --from=frontend /app/package.json ./frontend/package.json
EXPOSE 8000
CMD ["uvicorn", "backend.analytics_engine:app", "--host", "0.0.0.0", "--port", "8000"]
