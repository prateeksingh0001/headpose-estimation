from typing import List, Callable, Tuple

import tensorflow as tf
from tensorflow.layers import Dense

from headpose_estimation.models.base import BaseModel
from headpose_estimation.utils.utils import fn_from_str


class EulerAnglesPredictionHead(BaseModel):
    """
    Predicts Euler angles for yaw, pitch and roll from an image representation.

    Uses the tensorflow v1 Dense layer, this layer does not need the input dimensions it calculates them dynamically
    when the variables are initialized(with `session.run(tf.global_variables_initializer())`). At this time Tensorflow
    builds the static computation graph looks at the shape of the input tensor and uses it to create the weight matrix
    for the Dense layer.
    """

    def __init__(
        self,
        input_image_representation_tensor: tf.Tensor,
        layer_sizes: List[int],
        activation_function: str = "tf.nn.relu",
        session: tf.Session = None,
    ):
        """
        """
        self.input_image_representation_tensor = input_image_representation_tensor
        activation_fn = fn_from_str(activation_function)
        self.yaw_angle_prediction = self.intialize_weights_for_angle(
            input_img_representation_tensor=self.input_image_representation_tensor,
            layer_sizes=layer_sizes,
            name="yaw_predictor",
            activation_function=activation_fn
        )

        self.pitch_angle_prediction = self.intialize_weights_for_angle(
            input_img_representation_tensor=self.input_image_representation_tensor,
            layer_sizes=layer_sizes,
            name="pitch_predictor",
            activation_function=activation_fn
        )

        self.roll_angle_prediction = self.intialize_weights_for_angle(
            input_img_representation_tensor=self.input_image_representation_tensor,
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
        
        return input_img_representation_tensor

    def forward(self, input_tensor: tf.Tensor) -> Tuple[float, float, float]:
        yaw_angle, pitch_angle, roll_angle = self.session.run(
            [self.yaw_angle_prediction, self.pitch_angle_prediction, self.roll_angle_prediction],
            feed_dict = {
                self.input_image_representation_tensor: input_tensor
            }
        )
        
        return yaw_angle, pitch_angle, roll_angle
