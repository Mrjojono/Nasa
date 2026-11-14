# API REST : reste allumée et recharge out/ quand un entraînement se termine.
FROM mirror.gcr.io/library/python:3.11-slim

WORKDIR /opt/project
RUN pip install --no-cache-dir \
      fastapi==0.115.6 \
      "uvicorn[standard]==0.34.0" \
      pandas==2.2.3 \
      numpy==2.1.3 \
      scikit-learn==1.5.2 \
      joblib==1.4.2 \
      prometheus-fastapi-instrumentator==7.0.2 \
 && pip install --no-cache-dir --no-deps xgboost==2.1.3 \
 && pip install --no-cache-dir psycopg2-binary==2.9.10

ENV PLUIE_OUT=/opt/project/donnees_pluie_prevue/out
EXPOSE 8000
CMD ["python", "-u", "api.py", "--host", "0.0.0.0", "--port", "8000"]
