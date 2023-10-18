from abc import abstractmethod

import tensorflow as tf

class BaseModel:

    def predict(self, *arg, **kwargs) -> tf.Tensor:
        raise NotImplementedError(f"method predict for not implemented for {self.__class__.__name__}")

    @abstractmethod
    def create_model_graph(self):
        raise NotImplementedError(f"method create_train_graph is not implemented for {self.__class__.__name__}")
