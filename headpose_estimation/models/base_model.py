from typing import Tuple, Union

from abc import abstractmethod

import tensorflow as tf

class BaseModel:

    def forward(self, input_data: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError(f"forward for not implemented for {self.__class__.__name__}")

    @abstractmethod
    def create_model_graph(
        self,
        graph_def_path: str,
        input_node_name: tf.Tensor,
        output_node_name: tf.Tensor
    ) -> Tuple[Union[tf.Operation, tf.Tensor], Union[tf.Operation, tf.Tensor]]:
        raise NotImplementedError(f"method create_train_graph is not implemented for {self.__class__.__name__}")
