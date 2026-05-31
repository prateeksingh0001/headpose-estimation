import random
from importlib import import_module
from typing import Callable, Tuple, Type

import numpy as np
import tensorflow as tf
from tensorflow.train import Optimizer

from headpose_estimation.schema import OptimizerConfig


def fn_from_str(fn_module_path: str) -> Callable:
    """
    Returns a python callable from its full reference path.
    """
    module_name = ".".join(fn_module_path.split(".")[:-1])
    fn_name = fn_module_path.split(".")[-1]
    return getattr(import_module(module_name), fn_name)


def cls_from_str(cls_reference_path: str) -> Type:
    module, klass = cls_reference_path.rsplit(".", maxsplit=1)
    module = import_module(module)
    return getattr(module, klass)


def optimizer_factory(
    optimizer_config: OptimizerConfig, global_step=tf.Variable
) -> Tuple[Optimizer, tf.Tensor]:

    lr_decay = None
    if optimizer_config.learning_rate_decay_function:
        lr_decay_fn = fn_from_str(optimizer_config.learning_rate_decay_function)
        lr_decay_params = {
            "learning_rate": optimizer_config.learning_rate,
            "global_step": global_step,
            "decay_steps": optimizer_config.decay_steps,
            "decay_rate": optimizer_config.decay_rate,
            "name": "learning_rate",
        }
        lr_decay = lr_decay_fn(**lr_decay_params)

    optimizer_class = cls_from_str(cls_reference_path=optimizer_config.optimizer_class)
    optimizer_params = {
        "learning_rate": lr_decay
        if lr_decay is not None
        else optimizer_config.learning_rate,
        **optimizer_config.optimizer_params,
    }
    optimizer = optimizer_class(**optimizer_params)

    if lr_decay is None:
        lr_decay = tf.constant(optimizer_config.learning_rate, name="learning_rate")

    return optimizer, lr_decay


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


def set_global_seed(seed: int) -> None:
    """Sets a global level seed for random, numpy and tensorflow

    Args:
        seed (int): seed
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
