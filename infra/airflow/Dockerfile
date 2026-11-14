FROM apache/airflow:2.10.5-python3.11

USER airflow
COPY requirements.txt /tmp/requirements.txt
# --no-deps sur xgboost : évite nvidia-nccl (~centaines de Mo, inutile en CPU)
RUN pip install --no-cache-dir \
      pandas==2.2.3 \
      numpy==2.1.3 \
      scikit-learn==1.5.2 \
      joblib==1.4.2 \
      prometheus-client==0.21.1 \
      scipy \
      threadpoolctl \
      rasterio \
      netCDF4 \
 && pip install --no-cache-dir --no-deps xgboost==2.1.3
