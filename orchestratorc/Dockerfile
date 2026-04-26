FROM python:3.11-slim@sha256:a1580d0e5ce8b1afab9fd4b0e8e7b0b9f8e8c0e8e8e8e8e8e8e8e8e8e8e8e8

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
