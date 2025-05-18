import numpy as np
import tensorflow as tf
from keras.src.saving import load_model
from PIL import Image
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from ml import IMAGES_DIR, IMG_HEIGHT, IMG_WIDTH, MODEL_PATH


class ImageRecognizer:
    def __init__(self, model: str = None):
        self.model = model

    @classmethod
    def load_model(cls, model_path: str = MODEL_PATH) -> "ImageRecognizer":
        model = load_model(model_path)
        return cls(model)

    def build(self):
        batch_size = 8
        channel = 1
        cropped_height = IMG_HEIGHT // 4  # Nowa wysokość po przycięciu

        # Funkcja do przycinania nagłówka
        def crop_header(image, label):
            return tf.image.crop_to_bounding_box(
                image, 0, 0, cropped_height, IMG_WIDTH
            ), label

        # Ładowanie danych bez użycia .load()
        train_ds = tf.keras.utils.image_dataset_from_directory(
            IMAGES_DIR,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            color_mode="grayscale",
            batch_size=batch_size,
            shuffle=True,
            seed=42,
            validation_split=0.2,
            subset="training",
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            IMAGES_DIR,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            color_mode="grayscale",
            batch_size=batch_size,
            shuffle=True,
            seed=42,
            validation_split=0.2,
            subset="validation",
        )

        # Przycinanie i optymalizacja datasetów
        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.map(crop_header, num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.map(crop_header, num_parallel_calls=AUTOTUNE)

        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Reszta kodu pozostaje bez zmian...
        model = Sequential([
            Conv2D(
                32,
                (5, 5),
                activation="relu",
                input_shape=(cropped_height, IMG_WIDTH, channel),
            ),
            MaxPooling2D((3, 3)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation="relu"),
            GlobalAveragePooling2D(),
            Dense(512, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ])

        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5),
                tf.keras.callbacks.ModelCheckpoint("nowy.keras", save_best_only=True),
            ],
        )

        self.model = model

    def recognize(self, img: Image):
        if self.model.input_shape[-1] == 1:
            img = img.convert("L")

        width, height = img.size
        cropped_height = height // 4
        img = img.crop((0, 0, width, cropped_height))

        img = img.resize((self.model.input_shape[2], self.model.input_shape[1]))

        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0

        if len(img_array.shape) == 2:  # Dla obrazów w skali szarości
            img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)
        probability = prediction[0][0]

        threshold = 0.5
        result = {
            "is_wz": probability >= threshold,
            "confidence": probability if probability >= threshold else 1 - probability,
            "class": "WZ" if probability >= threshold else "NO_WZ",
        }

        return result


if __name__ == "__main__":
    ImageRecognizer().build()
