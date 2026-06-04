from typing import Dict, List

import numpy as np
import tensorflow as tf

from headpose_estimation.models.angle_prediction_heads import EulerAnglesPredictionHead
from headpose_estimation.models.base import BaseModel
from headpose_estimation.models.pretrained_image_model import (
  PretrainedBackBoneImageModel,
)


class EulerAnglesPredictionModel(BaseModel[str, Dict[str, float]]):
  """
  Container class for a base pretrained image backbone model and a prediction head.
  Allows providing image data and getting the angle predictions directly from it.
  """

  def __init__(
    self,
    backbone_image_model: PretrainedBackBoneImageModel,
    euler_angle_prediction_head: EulerAnglesPredictionHead,
  ) -> None:

    self._backbone_image_model = backbone_image_model
    self._angle_prediction_head = euler_angle_prediction_head

  @property
  def input_placeholder(self) -> tf.Tensor:
    return self._backbone_image_model.input_placeholder

  @property
  def prediction_ops(self) -> Dict[str, tf.Tensor]:
    return self._angle_predictin_head.prediction_ops

  def predict(self, session: tf.compat.v1.Session, prediction_input: List[str]) -> List[Dict[str, float]]:
    backbone_model_representations: List[np.ndarray] = self._backbone_image_model.predict(
      session=session, prediction_input=prediction_input
    )

    output_angles = self.angle_prediction_head.predict(session=session, prediction_input=backbone_model_representations)

    return output_angles
