FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md requirements.lock ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.lock && pip install --no-cache-dir --no-deps . && useradd --create-home appuser
COPY artifacts/router.joblib ./artifacts/router.joblib
USER appuser
EXPOSE 8001
CMD ["uvicorn", "intent_router.api:app", "--host", "0.0.0.0", "--port", "8001"]
