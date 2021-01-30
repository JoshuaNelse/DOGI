import tensorflow as tf
import tensorflow_hub as hub
from modules.service.image_service import ImageProcessor as ip


class DOGI:
    __MODEL_FILE__ = 'models/20210129-01401611884422-full-data-model.h5'
    __loaded_model__ = None

    @staticmethod
    def load_model():
        DOGI.__loaded_model__ = tf.keras.models.load_model(
            DOGI.__MODEL_FILE__, custom_objects={'KerasLayer': hub.KerasLayer})

    @staticmethod
    def get_model():
        if DOGI.__loaded_model__ is None:
            DOGI.load_model()
        return DOGI.__loaded_model__

    @staticmethod
    def batch_for_prediction(image_tensor, batch_size=32):
        data = tf.data.Dataset.from_tensors(image_tensor)
        data_batch = data.batch(batch_size)
        return data_batch
