from abc import abstractmethod

import tensorflow as tf

class BaseModel:

    def forward(self, session: tf.Session, input_data: tf.Tensor, *arg, **kwargs) -> tf.Tensor:
        raise NotImplementedError(f"forward for not implemented for {self.__class__.__name__}")

    @abstractmethod
    def create_model_graph(self, model_dir: str):
        raise NotImplementedError(f"method create_train_graph is not implemented for {self.__class__.__name__}")
