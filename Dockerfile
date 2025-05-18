FROM python:3.12-slim

WORKDIR /app
COPY . /app

ENV CUDA_VISIBLE_DEVICES="0"
ENV HF_HOME="/app/model_cache/huggingface"
ENV TRANSFORMERS_CACHE="/app/model_cache/transformers"
ENV TORCH_HOME="/app/model_cache/torch"
ENV TFHUB_CACHE_DIR="/app/model_cache/tensorflow"
ENV KERAS_HOME="/app/model_cache/keras"
ENV EASYOCR_MODULE_PATH="/app/model_cache/easyocr"

RUN apt update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    poppler-utils 

RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install tensorflow-cpu easyocr pdf2image pytesseract


RUN python -c "import easyocr; easyocr.Reader(['pl', 'en'], gpu=False)"

CMD ["python", "main.py", "--path", "/app/input", "--output", "/app/output"]
