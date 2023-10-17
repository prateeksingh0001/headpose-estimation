
from abc import abstractmethod

import tensorflow as tf

class BaseModel:

    @abstractmethod
    def predict(self, *arg, **kwargs) -> tf.Tensor:
        raise NotImplementedError(f"method predict for not implemented for {self.__class__.__name__}")
