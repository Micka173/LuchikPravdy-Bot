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

RUN pip install gdown
RUN wget -O chroma_db.zip "https://limewire.com/d/GVgrI#oVwr131aTX"
RUN unzip chroma_db.zip && rm chroma_db.zip

CMD ["python", "bot.py"]
