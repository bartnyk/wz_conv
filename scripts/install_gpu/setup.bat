@echo off

IF NOT EXIST .venv (
    python -m venv .venv
)

call .venv\Scripts\activate
pip install --upgrade pip
pip install pytorch tensorflow-gpu easyocr pdf2image pytesseract
