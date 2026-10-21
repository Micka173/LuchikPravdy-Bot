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

RUN apt-get update && apt-get install -y wget unzip
RUN wget -O chroma_db.zip "https://drive.google.com/file/d/14K_86NOZWaSs2hUPQxY1Wny3yACUZklf/view?usp=sharing" && unzip chroma_db.zip && rm chroma_db.zip

CMD ["python", "bot.py"]
