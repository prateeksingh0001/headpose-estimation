from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import logging
from pathlib import Path

import tensorflow as tf
from tensorflow.python.platform import gfile

from headpose_estimation.utils.tensorflow_model_handler import TensorflowV1ModelConfig


logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Baseclass for all the models used in this project
    """

    def __init__(
        self,
        session: tf.Session,
        **kwargs
    ):
        self.session = session


class BasePretrainedBackBoneImageModel(BaseModel):
    """
    Baseclass for all the pretrained backbone image models.
    """

    def __init__(
        self,
        tf_model_config: TensorflowV1ModelConfig,
        session: tf.Session = None
    ):

        self.pretrained_img_model_input_node, self.pretrained_img_model_output_node = self._create_model_graph(
            graph_def_path=tf_model_config.graph_definition_path,
            input_node_name=tf_model_config.input_node_name,
            output_node_name=tf_model_config.output_node_name
        )

        self.session = session

    @abstractmethod
    def _preprocess(
        *args,
        **kwargs,
    ) -> tf.Tensor:
        pass

    def _create_model_graph(
        self,
        graph_def_path: str,
        input_node_name: str,
        output_node_name: str
    ) -> Tuple[Union[tf.Operation, tf.Tensor], Union[tf.Operation, tf.Tensor]]:

        graph_def = self._load(graph_def_path)

        input_tensor, output_tensor = tf.import_graph_def(
            graph_def,
            return_elements=[input_node_name, output_node_name]
        )

        return input_tensor, output_tensor

    @staticmethod
    def _load(graph_def_path: Union[str, Path]):
        with gfile.FastGFile(graph_def_path, "rb") as f_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f_handle.read())

            return graph_def
    
    @property
    def output_node(self) -> tf.Tensor:
        return self.pretrained_img_model_output_node

    @abstractmethod
    def forward(self, input_data: List[bytes]) -> List[tf.Tensor]:
        raise NotImplementedError(f"forward for not implemented for {self.__class__.__name__}")
