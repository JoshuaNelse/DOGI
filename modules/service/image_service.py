import tensorflow as tf


class ImageProcessor:

    @staticmethod
    def process_image(image_in, image_size=224):
        """
        input file_path to image and turns target image into a tensor
        default image size is 224x224
        """
        # Turn the jpeg image into a numerical tensor with 3 color channels
        image = tf.image.decode_jpeg(image_in.read(), channels=3)
        # Convert the color channel values from 0-255 to 0-1
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Resize the image to our desired value 224x224
        image = tf.image.resize(image, size=[image_size, image_size])
        print(image.shape)
        return image
