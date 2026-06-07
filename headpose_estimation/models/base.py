from abc import ABC, abstractmethod
from typing import Dict, Generic, List, TypeVar

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

from headpose_estimation.utils.constants import (
  EULER_ANGLE_NORMALIZATION_FACTOR,
  PITCH_ANGLE_KEY,
  ROLL_ANGLE_KEY,
  YAW_ANGLE_KEY,
)

ModelInput = TypeVar("ModelInput")
PredictionOutput = TypeVar("PredictionOutput")


class BaseModel(Generic[ModelInput, PredictionOutput], ABC):
  """
  Baseclass for all the models used in this project
  """

  @property
  @abstractmethod
  def prediction_ops(self) -> Dict[str, tf.Tensor]:
    pass

  @property
  @abstractmethod
  def input_placeholder(self) -> tf.Tensor:
    pass

  @abstractmethod
  def predict(self, session: tf.compat.v1.Session, prediction_input: List[ModelInput]) -> List[PredictionOutput]:
    pass


class BaseAnglePredictionHeadModel(BaseModel[np.ndarray, Dict[str, float]]):
  @property
  def ground_truth_placeholders(self) -> Dict[str, tf.Tensor]:
    return self._ground_truth_placeholder

  @property
  def training_ops(self) -> Dict[str, tf.Tensor]:
    return self._training_ops

  @property
  def loss_ops(self) -> Dict[str, tf.Tensor]:
    return self._loss_ops

  @property
  def summary_ops(self) -> List[tf.Tensor]:
    return self._summary_ops

  @property
  def prediction_ops(self) -> Dict[str, tf.Tensor]:
    return self._prediction_ops

  @property
  def input_placeholder(self) -> tf.Tensor:
    return self._input_placeholder

  def predict(self, session: tf.compat.v1.Session, prediction_input: List[np.ndarray]) -> List[Dict[str, float]]:
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
          YAW_ANGLE_KEY: yaw_angle[i][0] * EULER_ANGLE_NORMALIZATION_FACTOR,
          PITCH_ANGLE_KEY: pitch_angle[i][0] * EULER_ANGLE_NORMALIZATION_FACTOR,
          ROLL_ANGLE_KEY: roll_angle[i][0] * EULER_ANGLE_NORMALIZATION_FACTOR,
        }
      )

    return output

  def save_as_frozen_graph(self, session: tf.compat.v1.Session, output_graphdef_path: str) -> None:
    output_node_names = [x.op.name for x in self.prediction_ops.values()]
    frozen_graph_definition = graph_util.convert_variables_to_constants(
      session, session.graph.as_graph_def(), output_node_names
    )

    with tf.io.gfile.GFile(output_graphdef_path, "wb") as f:
      f.write(frozen_graph_definition.SerializeToString())
