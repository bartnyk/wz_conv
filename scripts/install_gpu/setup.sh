#!/bin/bash

# Sprawdzenie, czy środowisko już istnieje
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

source .venv/bin/activate
pip install --upgrade pip
pip install pytorch tensorflow-gpu easyocr pdf2image pytesseract
