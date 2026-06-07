import math
from typing import Callable, Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.layers import Dense
from tensorflow.losses import Reduction, mean_squared_error
from tensorflow.nn import softmax, softmax_cross_entropy_with_logits
from tensorflow.train import Optimizer
from typing_extensions import Literal

from headpose_estimation.models.base import BaseAnglePredictionHeadModel
from headpose_estimation.utils.constants import (
  EULER_ANGLE_NORMALIZATION_FACTOR,
  PITCH_ANGLE_KEY,
  ROLL_ANGLE_KEY,
  TOTAL,
  YAW_ANGLE_KEY,
)
from headpose_estimation.utils.utils import fn_from_str


class EulerAnglesPredictionHead(BaseAnglePredictionHeadModel):
  """
  Predicts Euler angles for yaw, pitch and roll from an image representation.

  Uses the tensorflow v1 Dense layer, this layer does not need the input dimensions it calculates them dynamically
  when the variables are initialized(with `session.run(tf.global_variables_initializer())`). At this time Tensorflow
  builds the static computation graph looks at the shape of the input tensor and uses it to create the weight matrix
  for the Dense layer.
  """

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
    yaw_angle_prediction = self._intialize_weights_for_an_angle(
      input_image_representation_tensor=self._input_placeholder,
      layer_sizes=layer_sizes,
      name="yaw_predictor",
      activation_function=activation_fn,
    )

    pitch_angle_prediction = self._intialize_weights_for_an_angle(
      input_image_representation_tensor=self._input_placeholder,
      layer_sizes=layer_sizes,
      name="pitch_predictor",
      activation_function=activation_fn,
    )

    roll_angle_prediction = self._intialize_weights_for_an_angle(
      input_image_representation_tensor=self._input_placeholder,
      layer_sizes=layer_sizes,
      name="roll_predictor",
      activation_function=activation_fn,
    )

    self._prediction_ops = {
      YAW_ANGLE_KEY: yaw_angle_prediction,
      PITCH_ANGLE_KEY: pitch_angle_prediction,
      ROLL_ANGLE_KEY: roll_angle_prediction,
    }

    self._learning_rate = learning_rate_tensor

    self._summary_ops = []

    self._initialize_training_ops(
      prediction_tensors=self._prediction_ops,
      optimizer=optimizer,
      global_step=global_step,
    )

  def _intialize_weights_for_an_angle(
    self,
    input_image_representation_tensor: tf.Tensor,
    layer_sizes: List[int],
    name: str,
    activation_function: Callable[[tf.Tensor], tf.Tensor],
  ) -> tf.Tensor:

    previous_layer_output = input_image_representation_tensor
    all_layer_sizes = [*layer_sizes, 1]
    with tf.name_scope(name):
      for i, output_size in enumerate(
        [*layer_sizes, 1]  # Appends 1 as the final output angle layer
      ):
        layer = Dense(
          units=output_size,
          use_bias=False,
          # The last layer should return raw logits and not ReLU'ed logits
          activation=activation_function if i < (len(all_layer_sizes) - 1) else None,
          trainable=True,
        )
        previous_layer_output = layer(previous_layer_output)

    return previous_layer_output

  def _initialize_training_ops(
    self,
    prediction_tensors: Dict[str, tf.Tensor],
    optimizer: Optimizer,
    global_step: tf.Variable,
  ) -> List[tf.Tensor]:
    yaw_ground_truth_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None], name="input_yaw_gt")
    roll_ground_truth_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None], name="input_roll_gt")
    pitch_ground_truth_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None], name="input_pitch_gt")

    self._ground_truth_placeholder = {
      YAW_ANGLE_KEY: yaw_ground_truth_placeholder,
      ROLL_ANGLE_KEY: roll_ground_truth_placeholder,
      PITCH_ANGLE_KEY: pitch_ground_truth_placeholder,
    }

    yaw_loss = mean_squared_error(
      yaw_ground_truth_placeholder,
      tf.squeeze(prediction_tensors[YAW_ANGLE_KEY], axis=1),
    )
    roll_loss = mean_squared_error(
      roll_ground_truth_placeholder,
      tf.squeeze(prediction_tensors[ROLL_ANGLE_KEY], axis=1),
    )
    pitch_loss = mean_squared_error(
      pitch_ground_truth_placeholder,
      tf.squeeze(prediction_tensors[PITCH_ANGLE_KEY], axis=1),
    )

    total_loss = yaw_loss + roll_loss + pitch_loss

    optimizer_node = optimizer.minimize(total_loss, global_step=global_step)

    # Summary ops for getting results on tensorboard
    self._summary_ops.append(
      tf.compat.v1.summary.merge(
        [
          tf.compat.v1.summary.scalar("training/learning_rate", self._learning_rate),
          tf.compat.v1.summary.scalar("training/loss/total", total_loss),
          tf.compat.v1.summary.scalar("training/loss/yaw", yaw_loss),
          tf.compat.v1.summary.scalar("training/loss/pitch", pitch_loss),
          tf.compat.v1.summary.scalar("training/loss/roll", roll_loss),
        ]
      )
    )

    self._loss_ops = {
      TOTAL: total_loss,
      YAW_ANGLE_KEY: yaw_loss,
      ROLL_ANGLE_KEY: roll_loss,
      PITCH_ANGLE_KEY: pitch_loss,
    }

    self._training_ops = {
      "optimizer_node": optimizer_node,
    }

  def train(
    self,
    session: tf.compat.v1.Session,
    image_representations: List[np.ndarray],
    ground_truths: Dict[Literal["yaw", "pitch", "roll"], List[float]],
  ) -> List[float]:

    # Normalize ground truths
    ground_truths[YAW_ANGLE_KEY] = [x / EULER_ANGLE_NORMALIZATION_FACTOR for x in ground_truths[YAW_ANGLE_KEY]]
    ground_truths[PITCH_ANGLE_KEY] = [x / EULER_ANGLE_NORMALIZATION_FACTOR for x in ground_truths[PITCH_ANGLE_KEY]]
    ground_truths[ROLL_ANGLE_KEY] = [x / EULER_ANGLE_NORMALIZATION_FACTOR for x in ground_truths[ROLL_ANGLE_KEY]]

    batched_image_representations = np.vstack(image_representations)
    _, total_loss, yaw_loss, roll_loss, pitch_loss, summary = session.run(
      [
        *list(self.training_ops.values()),
        *list(self.loss_ops.values()),
        *self.summary_ops,
      ],
      feed_dict={
        self.input_placeholder: batched_image_representations,
        self.ground_truth_placeholders[YAW_ANGLE_KEY]: ground_truths[YAW_ANGLE_KEY],
        self.ground_truth_placeholders[PITCH_ANGLE_KEY]: ground_truths[PITCH_ANGLE_KEY],
        self.ground_truth_placeholders[ROLL_ANGLE_KEY]: ground_truths[ROLL_ANGLE_KEY],
      },
    )

    return total_loss, yaw_loss, roll_loss, pitch_loss, summary


class HopeNetPredictionHead(BaseAnglePredictionHeadModel):
  """
  Euler angles prediction head based on the HopeNet paper: https://arxiv.org/pdf/1710.00925
  """

  NUM_ANGLE_BUCKETS = 67  # 67 because the buckets range from [0, 66] both inclusive
  MAX_EULER_ANGLE = 99
  ANGLE_PER_BUCKET = 3
  ALPHA = 1
  YAW_CLASS_KEY = "yaw_class"
  PITCH_CLASS_KEY = "pitch_class"
  ROLL_CLASS_KEY = "roll_class"

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
    activation_function = fn_from_str(activation_function)

    yaw_angle_logits, yaw_angle_prediction = self._initialize_weights_for_angle_prediction(
      input_image_representation_tensor=self._input_placeholder,
      layer_sizes=layer_sizes,
      name="yaw_predictor",
      activation_function=activation_function,
    )

    pitch_angle_logits, pitch_angle_prediction = self._initialize_weights_for_angle_prediction(
      input_image_representation_tensor=self._input_placeholder,
      layer_sizes=layer_sizes,
      name="pitch_predictor",
      activation_function=activation_function,
    )

    roll_angle_logits, roll_angle_prediction = self._initialize_weights_for_angle_prediction(
      input_image_representation_tensor=self._input_placeholder,
      layer_sizes=layer_sizes,
      name="roll_predictor",
      activation_function=activation_function,
    )

    self._prediction_ops = {
      YAW_ANGLE_KEY: yaw_angle_prediction,
      PITCH_ANGLE_KEY: pitch_angle_prediction,
      ROLL_ANGLE_KEY: roll_angle_prediction,
    }

    self._learning_rate = learning_rate_tensor

    self._summary_ops = []

    self._initialize_training_ops(
      prediction_tensors=self._prediction_ops,
      softmax_logit_tensors={
        YAW_ANGLE_KEY: yaw_angle_logits,
        PITCH_ANGLE_KEY: pitch_angle_logits,
        ROLL_ANGLE_KEY: roll_angle_logits,
      },
      optimizer=optimizer,
      global_step=global_step,
    )

  def _initialize_weights_for_angle_prediction(
    self,
    input_image_representation_tensor: tf.Tensor,
    layer_sizes: List[int],
    name: str,
    activation_function: Callable[[tf.Tensor], tf.Tensor],
  ) -> tf.Tensor:
    previous_layer_output = input_image_representation_tensor
    all_layer_size = [*layer_sizes, self.NUM_ANGLE_BUCKETS]
    with tf.name_scope(name):
      for i, output_size in enumerate(all_layer_size):
        layer = Dense(
          units=output_size,
          use_bias=False,
          # The last layer should return raw logits and not ReLU'ed logits
          activation=activation_function if i < (len(all_layer_size) - 1) else None,
          trainable=True,
        )
        previous_layer_output = layer(previous_layer_output)

      softmax_logits = softmax(previous_layer_output, axis=-1, name=f"{name}/softmax_output")
      angle_prediction = (
        tf.reduce_sum(
          softmax_logits * tf.cast(tf.range(start=0, limit=67), tf.float32),
          axis=-1,
          keepdims=True,
        )
        * self.ANGLE_PER_BUCKET
      ) - self.MAX_EULER_ANGLE

    return previous_layer_output, angle_prediction

  def _calculate_loss_for_angle(
    self,
    angle_prediction_tensor: tf.Tensor,
    logit_tensor: tf.Tensor,
    angle_gt_tensor: tf.Tensor,
    angle_class_gt_tensor: tf.Tensor,
  ) -> tf.Tensor:
    angle_bucket_loss = softmax_cross_entropy_with_logits(labels=angle_class_gt_tensor, logits=logit_tensor)
    angle_prediction_loss = mean_squared_error(
      angle_gt_tensor,
      tf.squeeze(angle_prediction_tensor, axis=1),
      reduction=Reduction.NONE,
    )
    total_angle_loss = angle_bucket_loss + (self.ALPHA * angle_prediction_loss)
    return total_angle_loss

  def _initialize_training_ops(
    self,
    prediction_tensors: Dict[str, tf.Tensor],
    softmax_logit_tensors: Dict[str, tf.Tensor],
    optimizer: Optimizer,
    global_step: tf.Variable,
  ) -> List[tf.Tensor]:
    yaw_angle_ground_truth_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None], name="input_yaw_gt")
    yaw_class_gt_placeholder = tf.compat.v1.placeholder(
      tf.float32, shape=[None, self.NUM_ANGLE_BUCKETS], name="input_yaw_bucket_gt"
    )
    yaw_loss_tensor = self._calculate_loss_for_angle(
      angle_prediction_tensor=prediction_tensors[YAW_ANGLE_KEY],
      logit_tensor=softmax_logit_tensors[YAW_ANGLE_KEY],
      angle_gt_tensor=yaw_angle_ground_truth_placeholder,
      angle_class_gt_tensor=yaw_class_gt_placeholder,
    )

    roll_angle_ground_truth_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None], name="input_roll_gt")
    roll_class_gt_placeholder = tf.compat.v1.placeholder(
      tf.float32,
      shape=[None, self.NUM_ANGLE_BUCKETS],
      name="input_roll_bucket_gt",
    )
    roll_loss_tensor = self._calculate_loss_for_angle(
      angle_prediction_tensor=prediction_tensors[ROLL_ANGLE_KEY],
      logit_tensor=softmax_logit_tensors[ROLL_ANGLE_KEY],
      angle_gt_tensor=roll_angle_ground_truth_placeholder,
      angle_class_gt_tensor=roll_class_gt_placeholder,
    )

    pitch_angle_ground_truth_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None], name="input_pitch_gt")
    pitch_class_gt_placeholder = tf.compat.v1.placeholder(
      tf.float32,
      shape=[None, self.NUM_ANGLE_BUCKETS],
      name="input_pitch_bucket_gt",
    )
    pitch_loss_tensor = self._calculate_loss_for_angle(
      angle_prediction_tensor=prediction_tensors[PITCH_ANGLE_KEY],
      logit_tensor=softmax_logit_tensors[PITCH_ANGLE_KEY],
      angle_gt_tensor=pitch_angle_ground_truth_placeholder,
      angle_class_gt_tensor=pitch_class_gt_placeholder,
    )

    self._ground_truth_placeholder = {
      YAW_ANGLE_KEY: yaw_angle_ground_truth_placeholder,
      self.YAW_CLASS_KEY: yaw_class_gt_placeholder,
      ROLL_ANGLE_KEY: roll_angle_ground_truth_placeholder,
      self.ROLL_CLASS_KEY: roll_class_gt_placeholder,
      PITCH_ANGLE_KEY: pitch_angle_ground_truth_placeholder,
      self.PITCH_CLASS_KEY: pitch_class_gt_placeholder,
    }

    total_loss_tensor = yaw_loss_tensor + pitch_loss_tensor + roll_loss_tensor

    total_loss = tf.reduce_mean(total_loss_tensor)

    optimizer_node = optimizer.minimize(total_loss, global_step=global_step)

    yaw_loss = tf.reduce_mean(yaw_loss_tensor)
    pitch_loss = tf.reduce_mean(pitch_loss_tensor)
    roll_loss = tf.reduce_mean(roll_loss_tensor)

    self._summary_ops.append(
      tf.compat.v1.summary.merge(
        [
          tf.compat.v1.summary.scalar(
            "training/learning_rate",
            self._learning_rate,
          ),
          tf.compat.v1.summary.scalar("training/total_loss", total_loss),
          tf.compat.v1.summary.scalar("training/yaw_loss", yaw_loss),
          tf.compat.v1.summary.scalar("training/pitch_loss", pitch_loss),
          tf.compat.v1.summary.scalar("training/roll_loss", roll_loss),
        ]
      )
    )

    self._loss_ops = {
      TOTAL: total_loss,
      YAW_ANGLE_KEY: yaw_loss,
      ROLL_ANGLE_KEY: roll_loss,
      PITCH_ANGLE_KEY: pitch_loss,
    }

    self._training_ops = {"optimizer_node": optimizer_node}

  def train(
    self,
    session: tf.compat.v1.Session,
    image_representations: List[np.ndarray],
    ground_truths: Dict[Literal["yaw", "pitch", "roll"], List[float]],
  ) -> List[float]:

    angle_bins = {
      self.YAW_CLASS_KEY: [
        math.floor((angle + self.MAX_EULER_ANGLE) / self.ANGLE_PER_BUCKET) for angle in ground_truths[YAW_ANGLE_KEY]
      ],
      self.PITCH_CLASS_KEY: [
        math.floor((angle + self.MAX_EULER_ANGLE) / self.ANGLE_PER_BUCKET) for angle in ground_truths[PITCH_ANGLE_KEY]
      ],
      self.ROLL_CLASS_KEY: [
        math.floor((angle + self.MAX_EULER_ANGLE) / self.ANGLE_PER_BUCKET) for angle in ground_truths[ROLL_ANGLE_KEY]
      ],
    }
    angle_bin_ground_truths = {
      self.YAW_CLASS_KEY: np.eye(self.NUM_ANGLE_BUCKETS)[angle_bins[self.YAW_CLASS_KEY]],
      self.PITCH_CLASS_KEY: np.eye(self.NUM_ANGLE_BUCKETS)[angle_bins[self.PITCH_CLASS_KEY]],
      self.ROLL_CLASS_KEY: np.eye(self.NUM_ANGLE_BUCKETS)[angle_bins[self.ROLL_CLASS_KEY]],
    }

    ground_truths = {
      YAW_ANGLE_KEY: [angle / EULER_ANGLE_NORMALIZATION_FACTOR for angle in ground_truths[YAW_ANGLE_KEY]],
      PITCH_ANGLE_KEY: [angle / EULER_ANGLE_NORMALIZATION_FACTOR for angle in ground_truths[PITCH_ANGLE_KEY]],
      ROLL_ANGLE_KEY: [angle / EULER_ANGLE_NORMALIZATION_FACTOR for angle in ground_truths[ROLL_ANGLE_KEY]],
    }

    batched_image_representations = np.vstack(image_representations)
    _, total_loss, yaw_loss, roll_loss, pitch_loss, summary = session.run(
      [
        *list(self.training_ops.values()),
        *list(self.loss_ops.values()),
        *self.summary_ops,
      ],
      feed_dict={
        self.input_placeholder: batched_image_representations,
        self.ground_truth_placeholders[YAW_ANGLE_KEY]: ground_truths[YAW_ANGLE_KEY],
        self.ground_truth_placeholders[PITCH_ANGLE_KEY]: ground_truths[PITCH_ANGLE_KEY],
        self.ground_truth_placeholders[ROLL_ANGLE_KEY]: ground_truths[ROLL_ANGLE_KEY],
        self.ground_truth_placeholders[self.YAW_CLASS_KEY]: angle_bin_ground_truths[self.YAW_CLASS_KEY],
        self.ground_truth_placeholders[self.PITCH_CLASS_KEY]: angle_bin_ground_truths[self.PITCH_CLASS_KEY],
        self.ground_truth_placeholders[self.ROLL_CLASS_KEY]: angle_bin_ground_truths[self.ROLL_CLASS_KEY],
      },
    )

    return total_loss, yaw_loss, roll_loss, pitch_loss, summary
