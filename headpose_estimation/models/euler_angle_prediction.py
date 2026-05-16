from typing import List, Callable, Optional, Tuple

import tensorflow as tf
from tensorflow.layers import Dense
import numpy as np

from headpose_estimation.models.base import BaseModel, BasePretrainedBackBoneImageModel
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
        activation_function: str = "tensorflow.nn.relu",
        session: tf.Session = None,
    ):

        self.input_image_representation_tensor = input_image_representation_tensor
        activation_fn = fn_from_str(activation_function)

        self.yaw_angle_prediction = self.intialize_weights_for_angle(
            input_image_representation_tensor=self.input_image_representation_tensor,
            layer_sizes=layer_sizes,
            name="yaw_predictor",
            activation_function=activation_fn
        )

        self.pitch_angle_prediction = self.intialize_weights_for_angle(
            input_image_representation_tensor=self.input_image_representation_tensor,
            layer_sizes=layer_sizes,
            name="pitch_predictor",
            activation_function=activation_fn
        )

        self.roll_angle_prediction = self.intialize_weights_for_angle(
            input_image_representation_tensor=self.input_image_representation_tensor,
            layer_sizes=layer_sizes,
            name="roll_predictor",
            activation_function=activation_fn
        )
        super().__init__(session=session)

    def intialize_weights_for_angle(
        self,
        input_image_representation_tensor: tf.Tensor,
        layer_sizes: List[int],
        name: str,
        activation_function: Callable[[tf.Tensor], tf.Tensor]
    ):
        previous_layer_output = input_image_representation_tensor
        with tf.name_scope(name):
            for i, output_size in enumerate(layer_sizes):
                layer = Dense(
                    units=output_size,
                    use_bias=False,
                    activation=activation_function,
                    trainable=True,
                )
                previous_layer_output = layer(previous_layer_output)

        return previous_layer_output

    def forward(self, input_tensor: np.ndarray) -> Tuple[float, float, float]:
        yaw_angle, pitch_angle, roll_angle = self.session.run(
            [self.yaw_angle_prediction, self.pitch_angle_prediction, self.roll_angle_prediction],
            feed_dict = {
                self.input_image_representation_tensor: input_tensor
            }
        )

        return yaw_angle, pitch_angle, roll_angle


class EulerAnglesPredictionModel(BaseModel):
    def __init__(
        self,
        backbone_image_model: BasePretrainedBackBoneImageModel,
        session: tf.Session,
        euler_angle_prediction_head: Optional[EulerAnglesPredictionHead] = None,
        **kwargs
    ):

        self.backbone_image_model = backbone_image_model
        self.angle_prediction_head = euler_angle_prediction_head

        if not euler_angle_prediction_head:
            euler_angle_prediction_head = EulerAnglesPredictionHead(
                input_image_representation_tensor=tf.placeholder(
                    tf.float32,
                    shape=[None, self.backbone_image_model.output_tensor_size],
                    name="angle_prediction_input"
                ),
                layer_sizes=[512, 1]
            )
        self.euler_angle_prediction_head = euler_angle_prediction_head
        super().__init__(session=session)

    def forward(self, images: List[bytes]) -> Tuple[float, float, float]:
        backbone_model_representation = self.backbone_image_model.forward(input_data=images)
        output_angles = self.angle_prediction_head.forward(input_tensor=backbone_model_representation)
        return output_angles
