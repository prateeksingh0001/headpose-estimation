from abc import ABC, abstractmethod
from typing import Tuple, Union

import tensorflow as tf

class BaseModel(ABC):
    @abstractmethod
    def forward(self, input_data: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError(f"forward for not implemented for {self.__class__.__name__}")
