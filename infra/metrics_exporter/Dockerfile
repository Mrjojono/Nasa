# mirror.gcr.io si Docker Hub CDN timeout
FROM mirror.gcr.io/library/python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir prometheus-client==0.21.1
COPY exporter.py /app/exporter.py

ENV EXPORTER_PORT=9101
EXPOSE 9101
CMD ["python", "-u", "/app/exporter.py"]
