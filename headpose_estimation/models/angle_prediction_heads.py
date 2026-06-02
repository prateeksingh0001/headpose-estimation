from typing import Callable, Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.layers import Dense
from tensorflow.losses import mean_squared_error
from tensorflow.python.framework import graph_util
from tensorflow.train import Optimizer
from typing_extensions import Literal

from headpose_estimation.models.base import BaseAnglePredictionHeadModel
from headpose_estimation.utils.utils import fn_from_str


class EulerAnglesPredictionHead(BaseAnglePredictionHeadModel):
    """
    Predicts Euler angles for yaw, pitch and roll from an image representation.

    Uses the tensorflow v1 Dense layer, this layer does not need the input dimensions it calculates them dynamically
    when the variables are initialized(with `session.run(tf.global_variables_initializer())`). At this time Tensorflow
    builds the static computation graph looks at the shape of the input tensor and uses it to create the weight matrix
    for the Dense layer.
    """

    YAW_ANGLE_KEY = "yaw"
    PITCH_ANGLE_KEY = "pitch"
    ROLL_ANGLE_KEY = "roll"

    # Since all the angles are in degrees we assume that the total range of angles will between -90 and 90 degrees
    # across each axis so we normalize the angles to be between [-1, 1].
    # This keeps the training stable as the difference between the predictions and ground truths are little and there
    # aren't wild swings in the MSE loss during training.
    EULER_ANGLE_NORMALIZATION_FACTOR = 90

    def __init__(
        self,
        input_representation_size: int,
        layer_sizes: List[int],
        optimizer: Optimizer,
        learning_rate_tensor: tf.Tensor,
        global_step: tf.Variable,
        activation_function: str = "tensorflow.nn.relu",
    ) -> None:

        self._input_placeholder = tf.compat.v1.placeholder(
            tf.float32,
            shape=[None, input_representation_size],
            name="input_image_representation",
        )

        activation_fn = fn_from_str(activation_function)
        yaw_angle_prediction = self.intialize_weights_for_an_angle(
            input_image_representation_tensor=self._input_placeholder,
            layer_sizes=layer_sizes,
            name="yaw_predictor",
            activation_function=activation_fn,
        )

        pitch_angle_prediction = self.intialize_weights_for_an_angle(
            input_image_representation_tensor=self._input_placeholder,
            layer_sizes=layer_sizes,
            name="pitch_predictor",
            activation_function=activation_fn,
        )

        roll_angle_prediction = self.intialize_weights_for_an_angle(
            input_image_representation_tensor=self._input_placeholder,
            layer_sizes=layer_sizes,
            name="roll_predictor",
            activation_function=activation_fn,
        )

        self._prediction_ops = {
            self.YAW_ANGLE_KEY: yaw_angle_prediction,
            self.PITCH_ANGLE_KEY: pitch_angle_prediction,
            self.ROLL_ANGLE_KEY: roll_angle_prediction,
        }

        self._learning_rate = learning_rate_tensor

        self.initialize_training_ops(
            prediction_tensors=self._prediction_ops,
            optimizer=optimizer,
            global_step=global_step,
        )

    def intialize_weights_for_an_angle(
        self,
        input_image_representation_tensor: tf.Tensor,
        layer_sizes: List[int],
        name: str,
        activation_function: Callable[[tf.Tensor], tf.Tensor],
    ) -> tf.Tensor:

        previous_layer_output = input_image_representation_tensor
        with tf.name_scope(name):
            for _, output_size in enumerate(
                [*layer_sizes, 1]  # Appends 1 as the final output angle layer
            ):
                layer = Dense(
                    units=output_size,
                    use_bias=False,
                    activation=activation_function,
                    trainable=True,
                )
                previous_layer_output = layer(previous_layer_output)

        return previous_layer_output

    def initialize_training_ops(
        self,
        prediction_tensors: Dict[str, tf.Tensor],
        optimizer: Optimizer,
        global_step: tf.Variable,
    ) -> List[tf.Tensor]:
        yaw_ground_truth_placeholder = tf.compat.v1.placeholder(
            tf.float32, shape=[None], name="input_yaw_gt"
        )
        roll_ground_truth_placeholder = tf.compat.v1.placeholder(
            tf.float32, shape=[None], name="input_roll_gt"
        )
        pitch_ground_truth_placeholder = tf.compat.v1.placeholder(
            tf.float32, shape=[None], name="input_pitch_gt"
        )

        self._ground_truth_placeholder = {
            self.YAW_ANGLE_KEY: yaw_ground_truth_placeholder,
            self.ROLL_ANGLE_KEY: roll_ground_truth_placeholder,
            self.PITCH_ANGLE_KEY: pitch_ground_truth_placeholder,
        }

        yaw_loss = mean_squared_error(
            yaw_ground_truth_placeholder,
            tf.squeeze(prediction_tensors[self.YAW_ANGLE_KEY], axis=1),
        )
        roll_loss = mean_squared_error(
            roll_ground_truth_placeholder,
            tf.squeeze(prediction_tensors[self.ROLL_ANGLE_KEY], axis=1),
        )
        pitch_loss = mean_squared_error(
            pitch_ground_truth_placeholder,
            tf.squeeze(prediction_tensors[self.PITCH_ANGLE_KEY], axis=1),
        )

        total_loss = yaw_loss + roll_loss + pitch_loss

        optimizer_node = optimizer.minimize(total_loss, global_step=global_step)

        # Summary ops for getting results on tensorboard
        self._summary_op = tf.compat.v1.summary.merge(
            [
                tf.compat.v1.summary.scalar(
                    "training/learning_rate", self._learning_rate
                ),
                tf.compat.v1.summary.scalar("training/loss/total", total_loss),
                tf.compat.v1.summary.scalar("training/loss/yaw", yaw_loss),
                tf.compat.v1.summary.scalar("training/loss/pitch", pitch_loss),
                tf.compat.v1.summary.scalar("training/loss/roll", roll_loss),
            ]
        )

        self._loss_ops = {
            "total": total_loss,
            self.YAW_ANGLE_KEY: yaw_loss,
            self.ROLL_ANGLE_KEY: roll_loss,
            self.PITCH_ANGLE_KEY: pitch_loss,
        }

        self._training_ops = {
            "optimizer_node": optimizer_node,
        }

    @property
    def prediction_ops(self) -> Dict[str, tf.Tensor]:
        return self._prediction_ops

    @property
    def input_placeholder(self) -> tf.Tensor:
        return self._input_placeholder

    @property
    def ground_truth_placeholders(self) -> Dict[str, tf.Tensor]:
        return self._ground_truth_placeholder

    @property
    def training_ops(self) -> Dict[str, tf.Tensor]:
        return self._training_ops

    @property
    def loss_ops(self) -> Dict[str, tf.Tensor]:
        return self._loss_ops

    def train(
        self,
        session: tf.compat.v1.Session,
        image_representations: List[np.ndarray],
        ground_truths: Dict[Literal["yaw", "pitch", "roll"], List[float]],
    ) -> List[float]:

        # Normalize ground truths
        ground_truths[self.YAW_ANGLE_KEY] = [
            x / self.EULER_ANGLE_NORMALIZATION_FACTOR
            for x in ground_truths[self.YAW_ANGLE_KEY]
        ]
        ground_truths[self.PITCH_ANGLE_KEY] = [
            x / self.EULER_ANGLE_NORMALIZATION_FACTOR
            for x in ground_truths[self.PITCH_ANGLE_KEY]
        ]
        ground_truths[self.ROLL_ANGLE_KEY] = [
            x / self.EULER_ANGLE_NORMALIZATION_FACTOR
            for x in ground_truths[self.ROLL_ANGLE_KEY]
        ]

        batched_image_representations = np.vstack(image_representations)
        _, total_loss, yaw_loss, roll_loss, pitch_loss, summary = session.run(
            [
                *list(self.training_ops.values()),
                *list(self.loss_ops.values()),
                self._summary_op,
            ],
            feed_dict={
                self.input_placeholder: batched_image_representations,
                self.ground_truth_placeholders[self.YAW_ANGLE_KEY]: ground_truths[
                    self.YAW_ANGLE_KEY
                ],
                self.ground_truth_placeholders[self.PITCH_ANGLE_KEY]: ground_truths[
                    self.PITCH_ANGLE_KEY
                ],
                self.ground_truth_placeholders[self.ROLL_ANGLE_KEY]: ground_truths[
                    self.ROLL_ANGLE_KEY
                ],
            },
        )

        return total_loss, yaw_loss, roll_loss, pitch_loss, summary

    def predict(
        self, session: tf.compat.v1.Session, prediction_input: List[np.ndarray]
    ) -> List[Dict[str, float]]:
        """Predict the Euler Angles using the image representations

        Args:
            image_input_representation (np.ndarray): Image representations of the shape
                                                     [Batch_size, 1, image_representation_size]

        Returns:
            np.ndarray: Predicted Euler angles(in the order of yaw, pitch, roll) of the shape [Batch_size, 3]
        """
        stacked_prediction_input = np.vstack(prediction_input)
        yaw_angle, pitch_angle, roll_angle = session.run(
            list(self.prediction_ops.values()),
            feed_dict={self.input_placeholder: stacked_prediction_input},
        )

        output: List[Dict[str, float]] = []
        for i in range(len(prediction_input)):
            output.append(
                {
                    self.YAW_ANGLE_KEY: yaw_angle[i][0]
                    * self.EULER_ANGLE_NORMALIZATION_FACTOR,
                    self.ROLL_ANGLE_KEY: roll_angle[i][0]
                    * self.EULER_ANGLE_NORMALIZATION_FACTOR,
                    self.PITCH_ANGLE_KEY: pitch_angle[i][0]
                    * self.EULER_ANGLE_NORMALIZATION_FACTOR,
                }
            )

        return output

    def save_as_frozen_graph(
        self, session: tf.compat.v1.Session, output_graphdef_path: str
    ) -> None:
        output_node_names = [x.op.name for x in self.prediction_ops.values()]
        frozen_graph_definition = graph_util.convert_variables_to_constants(
            session, session.graph.as_graph_def(), output_node_names
        )

        with tf.io.gfile.GFile(output_graphdef_path, "wb") as f:
            f.write(frozen_graph_definition.SerializeToString())
