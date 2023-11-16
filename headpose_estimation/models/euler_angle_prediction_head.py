from typing import List, Callable

import tensorflow as tf
from tensorflow.layers import Dense

from headpose_estimation.models.base_model import BaseModel
from headpose_estimation.utils.utils import fn_from_str


class EulerAnglePredictionHead:
    def __init__(
        self,
        input_image_representation_tensor: tf.Tensor,
        layer_sizes: List[int],
        activation_function: str = "tf.nn.relu"
    ):
        activation_fn = fn_from_str(activation_function)
        self.intialize_weights_for_angle(
            input_img_representation_tensor=input_image_representation_tensor,
            layer_sizes=layer_sizes,
            name="yaw_predictor",
            activation_function=activation_fn
        )
        self.intialize_weights_for_angle(
            input_img_representation_tensor=input_image_representation_tensor,
            layer_sizes=layer_sizes,
            name="pitch_predictor",
            activation_function=activation_fn
        )
        self.intialize_weights_for_angle(
            input_img_representation_tensor=input_image_representation_tensor,
            layer_sizes=layer_sizes,
            name="roll_predictor",
            activation_function=activation_fn
        )

    def intialize_weights_for_angle(
        self,
        input_img_representation_tensor,
        layer_sizes: List[int],
        name: str,
        activation_function: Callable[[tf.Tensor], tf.Tensor]
    ):
        with tf.name_scope(name):
            for i, output_size in enumerate(layer_sizes):
                input_img_representation_tensor = Dense(input_img_representation_tensor, output=output_size)
                if i != len(layer_sizes)-1:
                    input_img_representation_tensor = activation_function(input_img_representation_tensor)
