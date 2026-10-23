FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# xgboost needs the OpenMP runtime / o xgboost precisa do runtime OpenMP
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install dependencies first so Docker caches this layer
# instala as dependências primeiro para o Docker reaproveitar esta camada
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# default: predict the new proteins with the trained XGBoost model
# padrão: prevê as proteínas novas com o modelo XGBoost treinado
CMD ["python", "py/predict.py"]
