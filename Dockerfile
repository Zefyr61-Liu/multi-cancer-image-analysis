FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app
COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch>=2.5,<3' 'torchvision>=0.20,<1' \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY cancer_lab cancer_lab
COPY configs configs
COPY smoke_test.py .
ENTRYPOINT ["python", "-m", "cancer_lab"]
CMD ["--help"]
