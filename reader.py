import os
import re
from datetime import datetime
from shutil import move
from typing import List, Optional

import numpy as np
import pytesseract
import torch
from easyocr import Reader
from pdf2image import convert_from_path
from PIL import Image

from ml.core import ImageRecognizer

if tesseract_cmd := os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

if gpu_enabled := torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.4, 0)


poppler_path = os.getenv("POPPLER_PATH")

recognizer = ImageRecognizer.load_model()

reader = Reader(["en", "pl"], gpu=gpu_enabled)


class PdfImage:
    def __init__(self, image: Image) -> None:
        self.raw = image
        self._image = np.array(image)

    @property
    def obj(self) -> Image:
        return Image.fromarray(self._image)

    def get_content(self, psm: int = 6, oem: int = 3) -> str:
        allowed_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZK0123456789/-"

        return pytesseract.image_to_string(
            self._image,
            lang="eng",
            config=f"--psm {psm} --oem {oem} -c tessedit_char_whitelist={allowed_chars}",
        ).replace("\n", " ")

    def is_page_empty(self) -> bool:
        raw_image = np.array(self.raw)
        content = pytesseract.image_to_string(
            raw_image,
            lang="eng",
        )

        return len(content) < 100

    def cut(self) -> None:
        height, width = self._image.shape[:2]
        self._image = self._image[: int(height * 0.3) :]

    def save(self, *args, **kwargs):
        append_images = kwargs.pop("append_images", [])
        kwargs["append_images"] = [img.raw for img in append_images]
        return self.raw.save(*args, **kwargs)


class PdfFileProcessor:
    def __init__(
        self,
        file_path: str,
        output_dir: str = None,
    ) -> None:
        self.file_path: str = file_path

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File {self.file_path} not found")

        self.output_dir: str = self._ensure_dir_exists(
            output_dir or self.file_path.replace(".pdf", "")
        )
        self.done_dir: str = self._ensure_dir_exists(self.done_dir_path)
        self._images: List[PdfImage] = []
        self._wz_aggregation: dict[str, list[Image]] = {}
        self._processed: bool = False

    def _ensure_dir_exists(self, dir_path: str = None) -> str:
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    @property
    def done_dir_path(self) -> str:
        done_dir_name = datetime.now().strftime("%d-%m-%Y")
        return os.path.join(self.output_dir, done_dir_name)

    def _split_file(self) -> None:
        self._images = [
            PdfImage(img)
            for img in convert_from_path(
                self.file_path,
                poppler_path=poppler_path,
                dpi=200,
                thread_count=4,
                fmt="png",
            )
        ]

    def process_pdf(self) -> None:
        self._split_file()
        pattern = r"(WZK|WZ-\d+/\d+/[A-Z]+/\d+)"

        def find_number(content: str) -> Optional[str]:
            if wz_match := re.search(pattern, content):
                return wz_match.group(1)
            return None

        if not self._images:
            raise ValueError("PDF seems to be empty or broken.")

        wz_number: Optional[str] = None

        for i, image in enumerate(self._images):
            res = recognizer.recognize(image.obj)
            cls = res["class"]
            if cls == "WZ":
                wz_nr = find_number(image.get_content())

                if not wz_nr:
                    image.cut()
                    wz_nr = (
                        find_number(image.get_content())
                        or find_number(image.get_content(11, 3))
                        or find_number(image.get_content(6, 1))
                        or find_number(image.get_content(3, 3))
                    )

                if not wz_nr:
                    content = reader.readtext(
                        image._image,
                        decoder="beamsearch",
                        text_threshold=0.8,
                        low_text=0.3,
                        contrast_ths=0.2,
                        adjust_contrast=0.7,
                        canvas_size=4096,
                    )
                    text = " ".join([phrase for _, phrase, _ in content])
                    wz_nr = find_number(text)

                if wz_nr:
                    wz_number = wz_nr

            elif image.is_page_empty():
                continue

            if wz_number:
                try:
                    self._wz_aggregation[wz_number].append(image)
                except KeyError:
                    self._wz_aggregation[wz_number] = [image]

        self._processed = True

    def parse_filename(self, name: str) -> str:
        return name.replace("/", "_")

    def save_all(self) -> None:
        if not self._processed:
            raise ValueError("PDF not processed yet.")

        counter = 0

        for wz_number, images in self._wz_aggregation.items():
            file_path = f"{self.output_dir}/{self.parse_filename(wz_number)}.pdf"
            self.save_pdf(file_path, images)
            counter += 1

        print(f"Created {counter} new WZ's out of {self.file_path}.")
        self.move_done()

    def move_done(self) -> None:
        if not self._processed:
            raise ValueError("PDF not processed yet.")

        file_name = os.path.basename(self.file_path)
        new_file_path = os.path.join(self.done_dir_path, file_name)
        move(
            self.file_path,
            new_file_path,
        )
        print(f"Moved {file_name} to {new_file_path}.")

    def save_pdf(self, file_path: str, images: List[Image]) -> None:
        images[0].save(
            file_path,
            save_all=True,
            append_images=images[1:],
            force_update=True,
        )
