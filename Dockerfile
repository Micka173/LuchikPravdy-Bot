FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Устанавливаем wget, unzip и curl (для надежности)
RUN apt-get update && apt-get install -y wget unzip curl

# Скачиваем архив и сразу проверяем, что это действительно zip
RUN curl -L -o chroma_db.zip "https://limewire.com/d/GVgrI#oVwr131aTX" && \
    file chroma_db.zip && \
    unzip chroma_db.zip && \
    rm chroma_db.zip

CMD ["python", "bot.py"]
