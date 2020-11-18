import os
import tarfile
import tensorflow as tf


@tf.function
def load_image(buffer, width, height, scale, offset):
    img = tf.io.decode_bmp(buffer)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_with_pad(img, width, height)
    img = (img * scale) + offset
    img = tf.broadcast_to(img, tf.concat([tf.shape(img)[:-1], [3]], axis=0))
    return img

def get_serving_receiver(model, shape, scale, offset):
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def serving(input_image):
        img = tf.map_fn(lambda x: load_image(x, *shape, scale, offset), input_image, dtype=tf.float32)
        return model(img)
    return serving


def package_saved_model(saved_model_dir):
    saved_model_tar = os.path.join(saved_model_dir, 'saved_model.tar.gz')
    with tarfile.open(saved_model_tar, "w:gz") as tar:
        tar.add(saved_model_dir, arcname="saved_model")
    return saved_model_tar