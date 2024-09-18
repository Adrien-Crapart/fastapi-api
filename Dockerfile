# app/Dockerfile
FROM python:3.11-alpine

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY ./requirements.txt .

COPY * .

RUN pip install -U pip && \
  pip install -U -r /app/requirements.txt && \
  adduser --disabled-password --no-create-home data

USER data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]