# wz_conv

A project for recognition and processing of WZ documents (e.g., warehouse documents, invoices) using OCR, machine learning, and image/PDF processing tools.

## Requirements

- Python 3.12
- Linux (recommended)
- Docker (optional)
- Tesseract OCR
- Poppler-utils (pdf2image)
- Python libraries: tensorflow, torch, torchvision, easyocr, pdf2image, pytesseract

## Installation

### CPU Installation (Linux)

```bash
./scripts/install_cpu/setup.sh
```

### GPU Installation (Linux, with CUDA)

```bash
./scripts/install_gpu/setup.sh
```

## Usage

Main application:
```bash
python main.py --path /app/input --output /app/output
```

- Place input files in the `input/` directory
- Results will be saved in the `output/` directory

```bash
python main.py --path /app/input/my_file.pdf
```

- Run for single file `/app/input/my_file.pdf` 
- The output will be saved in the same directory as the input file

## Directory Structure

- `main.py` – main entry point
- `reader.py` – PDF/image file handling
- `ml/` – ML logic, parsers, models
- `model_cache/` – OCR model cache
- `output/` – program results
- `tests/` – unit tests and test data
- `scripts/install_cpu/` – setup scripts for CPU
- `scripts/install_gpu/` – setup scripts for GPU

## Environment Variables
- `TESSERACT_CMD` – Path to the Tesseract OCR executable
- `POPPLER_PATH` – Path to the Poppler package (e.g., `pdftoppm`)

## Notes

- The project defaults to CPU if no GPU is detected.
- External tools required: tesseract-ocr, poppler-utils.
