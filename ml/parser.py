import os
import random
import string
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from ml import IMG_HEIGHT, IMG_WIDTH


class ImageParser:
    previous_angle = 2
    previous_brightness_factor_index = 0
    previous_contrast_factor_index = 0

    def __init__(self, image_array: np.ndarray, path: str):
        self.original_image_array = image_array
        self.image_array = self.original_image_array.copy()
        self.path = path
        self.class_name = os.path.basename(os.path.dirname(self.path))

    def reset_image(self):
        self.image_array = self.original_image_array.copy()

    @property
    def operations(self) -> tuple:
        return (
            (self.resize,),
            (self.resize, self.add_gaussian_noise),
            (
                self.resize,
                self.add_gaussian_noise,
                self.enhance_contrast,
                self.blur_image,
            ),
            (
                self.resize,
                self.add_gaussian_noise,
                self.enhance_contrast,
                self.enhance_brightness,
            ),
            (
                self.resize,
                self.add_gaussian_noise,
                self.enhance_brightness,
            ),
            (
                self.resize,
                self.blur_image,
                self.enhance_contrast,
                self.enhance_brightness,
            ),
            (
                self.resize,
                self.rotate,
                self.blur_image,
            ),
            (self.resize, self.rotate),
            (self.resize, self.rotate, self.enhance_contrast),
            (self.resize, self.rotate, self.enhance_brightness),
            (
                self.resize,
                self.rotate,
                self.enhance_contrast,
                self.enhance_brightness,
            ),
            (
                self.resize,
                self.rotate,
                self.enhance_contrast,
                self.enhance_brightness,
            ),
            (
                self.resize,
                self.blur_image,
                self.enhance_contrast,
                self.rotate,
            ),
            (
                self.resize,
                self.blur_image,
                self.rotate,
                self.enhance_brightness,
            ),
            (
                self.resize,
                self.blur_image,
                self.rotate,
                self.enhance_brightness,
                self.enhance_contrast,
            ),
        )

    @classmethod
    def read_image(cls, image_path: str):
        image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return cls(image_array, image_path)

    def resize(self):
        self.image_array = cv2.resize(self.image_array, (IMG_WIDTH, IMG_HEIGHT))

    def blur_image(self):
        self.image_array = cv2.GaussianBlur(self.image_array, (1, 1), 0)
        self.image_array = cv2.equalizeHist(self.image_array)

    def add_gaussian_noise(self) -> np.ndarray:
        sigma = 0.01**0.5
        gaussian = np.random.normal(0, sigma, self.image_array.shape).astype(np.float32)
        self.image_array = self.image_array.astype(np.float32)
        self.image_array = cv2.addWeighted(self.image_array, 0.75, gaussian, 0.25, 0)
        self.image_array = np.clip(self.image_array, 0, 255).astype(np.uint8)

    def change_brightness(self, factor: float):
        if factor < 0.1 or factor > 1.9:
            raise ValueError("Factor must be between 0.1 and 1.9")

        img = Image.fromarray(self.image_array)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)
        self.image_array = np.array(img)

    def change_contrast(self, factor: float):
        if factor < 0.1 or factor > 1.9:
            raise ValueError("Factor must be between 0.1 and 1.9")

        img = Image.fromarray(self.image_array)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)
        self.image_array = np.array(img)

    def enhance_brightness(self):
        factors = [0.8, 1.2]
        current_factor = factors[self.previous_brightness_factor_index]
        self.change_brightness(current_factor)
        self.previous_brightness_factor_index ^= 1

    def enhance_contrast(self):
        factors = [0.8, 1.2]
        current_factor = factors[self.previous_contrast_factor_index]
        self.change_contrast(current_factor)
        self.previous_contrast_factor_index ^= 1

    def rotate_image(self, angle: int):
        if angle not in [-2, 2]:
            raise ValueError("Angle must be -2 or 2")

        img = Image.fromarray(self.image_array)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
        self.image_array = np.array(img)

    def rotate(self) -> None:
        self.rotate_image(self.previous_angle)
        self.previous_angle *= -1

    @classmethod
    def process_images_in_directory(cls, directory_path: str):
        """Przetwarza i augmentuje obrazy z katalogu wejściowego"""
        counter = 1
        dir_name = os.path.dirname(directory_path)

        for filename in os.listdir(directory_path):
            if filename.lower().endswith(".png"):
                input_path = os.path.join(directory_path, filename)
                image_parser = cls.read_image(input_path)

                for operation_set in image_parser.operations:
                    for operation in operation_set:
                        operation()

                    image_parser.save_as_new(f"{image_parser.class_name}_{counter}.png")
                    image_parser.reset_image()
                    counter += 1

                os.remove(input_path)

    def get_random_filename(self, length=20, extension=".png") -> str:
        letters = string.ascii_letters + string.digits
        random_string = "".join(random.choice(letters) for i in range(length))
        return random_string + extension

    def save(self):
        img = Image.fromarray(self.image_array)
        img.save(self.path)

    def save_as_new(self, filename: Optional[str] = None):
        if not filename:
            filename = self.get_random_filename()

        dir_path = os.path.dirname(self.path)
        new_path = os.path.join(dir_path, filename)

        img = Image.fromarray(self.image_array)
        img.save(new_path)


if __name__ == "__main__":
    ImageParser.process_images_in_directory("../images/NO_WZ")
    ImageParser.process_images_in_directory("../images/WZ")
