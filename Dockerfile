FROM python:3.11-slim
WORKDIR /app

COPY requirements-api.txt /app/
RUN pip install --no-cache-dir -r requirements-api.txt

COPY data/ /app/data/
COPY train_model.py app.py /app/
RUN mkdir -p /app/models /app/data /app/logs && python train_model.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
