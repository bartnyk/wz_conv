import os

import tensorflow as tf
from tensorflow.python.keras.mixed_precision.policy import set_global_policy

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


IMG_WIDTH, IMG_HEIGHT = 620, 877
IMAGES_DIR = os.path.join(ROOT_DIR, "images")
MODEL_PATH = os.path.join(ROOT_DIR, "WZ_model.keras")

gpus = tf.config.experimental.list_physical_devices("GPU")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)
#
# tf.keras.mixed_precision.set_global_policy("mixed_float16")
