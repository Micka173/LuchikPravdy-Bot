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
RUN gdown "https://drive.google.com/file/d/14K_86NOZWasS2hUPQxY1Wny3yACUZkF/view?usp=sharing" -O chroma_db.zip
RUN unzip chroma_db.zip && rm chroma_db.zip

CMD ["python", "bot.py"]
