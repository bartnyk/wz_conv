@echo off

IF NOT EXIST .venv (
    python -m venv .venv
)

call .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow-cpu easyocr pdf2image pytesseract
