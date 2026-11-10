FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models

EXPOSE 9696

CMD ["python", "-m", "uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "9696"]