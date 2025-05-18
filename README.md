# wz_conv

A project for recognition and processing of WZ documents (e.g., warehouse documents, invoices) using OCR, machine learning, and image/PDF processing tools.

## Requirements

- Python 3.12
- Docker (optional)
- Tesseract OCR
- Poppler-utils (pdf2image)
- Python libraries: tensorflow, torch, torchvision, easyocr, pdf2image, pytesseract

## Configuration

All main settings (such as watched folders, output paths, and processing options) are defined in `core/config.py`. Adjust this file to change input/output directories, model paths, or other parameters for your environment.

## Installation

### CPU Installation (Linux)

```bash
./scripts/install_cpu/setup.sh
```

### CPU Installation (Windows)

```bash
./scripts/install_cpu/setup.bat
```

### GPU Installation (Linux, with CUDA)

```bash
./scripts/install_gpu/setup.sh
```

### GPU Installation (Windows, with CUDA)

```bash
./scripts/install_gpu/setup.bat
```

### Docker

Build the image:
```bash
docker build -t wz_conv .
```

Run the container:
```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output wz_conv
```

> The container will automatically detect if there is no GPU and switch to CPU mode.

## Usage

- All configuration (input/output folders, model paths, etc.) is managed in `core/config.py`.
- The service will process all existing PDF files in the input folder at startup and will continue to monitor for new files.
- Each PDF page is processed in a separate thread for maximum performance.

### Manual Processing

To process files manually (one-off):
```bash
python main.py --path /path/to/input.pdf --output /path/to/output_directory
```
This will process the specified **PDF file** (or directory with **PDF files**) and save the results in the output directory.

### Real-time Processing
To run the service in real-time mode (watching for new files):
```bash
python service.py --path /path/to/input_directory --watch
```
This will monitor the specified directory for new files and process them as they arrive.


## Directory Structure

- `main.py` – main entry point (manual processing)
- `service.py` – daemon/service for background folder monitoring
- `core/config.py` – main configuration file (edit this to change paths and options)
- `core/` – core logic, handlers, file watching, and processing
- `core/ml/` – ML logic, parsers, models
- `scripts/install_cpu/` – setup scripts for CPU
- `scripts/install_gpu/` – setup scripts for GPU
- `WZ_model.keras` – Pre-trained model for WZ document recognition

## Notes

- The project defaults to CPU if no GPU is detected.
- External tools required: tesseract-ocr, poppler-utils.
- Example test files are located in `tests/testdata/`.
- All main paths and options are set in `core/config.py`.
