import os
import sys
from os import PathLike

import torch
from pydantic_settings import BaseSettings

ROOT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))


class Config(BaseSettings):
    GPU_ENABLED: bool

    TESSERACT_CMD: PathLike | None = None
    POPPLER_PATH: PathLike | None = None
    MODEL_PATH: PathLike = os.path.join(ROOT_DIR, "WZ_model.keras")

    IMG_WIDTH: int = 620
    IMG_HEIGHT: int = 877

    WZ_IMAGES_DIR_PATH: PathLike = os.path.join(ROOT_DIR, "wz_images")
    NOT_WZ_IMAGES_DIR_PATH: PathLike = os.path.join(ROOT_DIR, "wz_images")

    WATCHER_COOLDOWN: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        use_enum_values = True
        arbitrary_types_allowed = True

    def model_post_init(self, context):
        super().model_post_init(context)

        if self.GPU_ENABLED:
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.4, 0)
            else:
                raise RuntimeError(
                    "GPU unavailable. Please check your CUDA installation or set GPU_ENABLED to False."
                )
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
