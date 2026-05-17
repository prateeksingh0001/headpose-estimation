from importlib import import_module
from typing import Callable

import tensorflow as tf


def fn_from_str(fn_module_path: str) -> Callable:
    """
    Returns a python callable from its full reference path.
    """
    module_name = ".".join(fn_module_path.split(".")[:-1])
    fn_name = fn_module_path.split(".")[-1]
    return getattr(import_module(module_name), fn_name)


def add_jpeg_decoding(self):
    jpeg_data = tf.placeholder(tf.string, name="DecodeJPGInput")
    decoded_image = tf.image.decode_jpeg(
        jpeg_data, channels=self.model_info["input_depth"]
    )
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack(
        [self.model_info["input_height"], self.model_info["input_width"]]
    )
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
    offset_image = tf.subtract(resized_image, self.model_info["input_mean"])
    mul_image = tf.multiply(offset_image, 1.0 / self.model_info["input_std"])
    return jpeg_data, mul_image
