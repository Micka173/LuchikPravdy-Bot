FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-rus \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN wget -O chroma_db.zip "https://github.com/Micka173/LuchikPravdy-Bot/releases/download/v1/chroma_db.zip" && \
    unzip chroma_db.zip && \
    rm chroma_db.zip

COPY . .

CMD ["python", "bot.py"]
