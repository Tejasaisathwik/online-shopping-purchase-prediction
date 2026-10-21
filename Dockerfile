FROM python:3.13-slim

WORKDIR /app

COPY online_shopping_model.joblib .
COPY app.py .

RUN pip install --no-cache-dir pandas numpy scikit-learn joblib fastapi uvicorn

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
