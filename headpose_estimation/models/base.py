from abc import ABC, abstractmethod
from typing import Dict, Generic, List, TypeVar

import numpy as np
import tensorflow as tf

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
    def predict(
        self, session: tf.compat.v1.Session, prediction_input: List[ModelInput]
    ) -> List[PredictionOutput]:
        pass


class BaseAnglePredictionHeadModel(BaseModel[np.ndarray, Dict[str, float]]):
    @property
    @abstractmethod
    def training_ops(self) -> List[tf.Tensor]:
        pass

    @property
    @abstractmethod
    def ground_truth_placeholders(self) -> Dict[str, tf.Tensor]:
        pass

    @property
    @abstractmethod
    def loss_ops(self) -> Dict[str, tf.Tensor]:
        pass
