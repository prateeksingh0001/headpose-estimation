from typing import Tuple, Dict, Union, Any

import logging
import os
import sys
import tarfile
from pathlib import Path

from six.moves import urllib
import tensorflow as tf
from tensorflow.python.platform import gfile

from headpose_estimation.models.base_model import BaseModel


logger = logging.getLogger(__name__)


class TF1ModelInference(BaseModel):
    """The large pretrained backbone image model class.

    This class will contain the backbone image model which will be used to
    create the representations for the input image, which shall then be
    used as the input to the final layers for predicting the yaw, pitch
    and roll values.

    Attributes:
        architecture (str): name of the vision model architecture
        model_path (str): Path on the disk where the model is downloaded to.
                          If the model is already present at that path then
                          it's not downloaded again.
    """


    def __init__(
        self,
        graph_def_path: str,
        input_node_name: str,
        output_node_name: str,
        input_image_size: Tuple[int, int],
        input_image_depth: int,
        session: tf.Session = None
    ):
        self.pretrained_img_model_input_node, self.pretrained_img_model_output_node = self.create_model_graph(
            graph_def_path, input_node_name, output_node_name
        )

        self.input_node, self.resized_image_output_node = self.create_input_preparation_nodes(
            input_image_size, input_image_depth
        )

        self.session = session

    @staticmethod
    def create_input_preparation_nodes(
        pretrained_model_input_size: Tuple(int, int),
        input_image_depth: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        input_image_data = tf.placeholder(dtype=tf.float32)

        resized_image_shape = [None, pretrained_model_input_size[0], pretrained_model_input_size[1], input_image_depth]
        resized_image = tf.image.resize_bilinear(input_image_data, resized_image_shape)

        return input_image_data, resized_image

    def create_model_graph(
        self,
        graph_def_path: str,
        input_node_name: str,
        output_node_name: str
    ) -> Tuple[Union[tf.Operation, tf.Tensor], Union[tf.Operation, tf.Tensor]]:

        with gfile.FastGFile(graph_def_path, "rb") as f_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f_handle.read())

        input_tensor, output_tensor = tf.import_graph_def(
            graph_def,
            return_elements=[input_node_name,output_node_name]
        )

        return input_tensor, output_tensor

    def forward(self, input_data: tf.Tensor) -> tf.Tensor:
        resized_image = self.session.run(self.resized_image_output_node, feed_dict={self.input_node: input_data})
        image_representation = self.session.run(
            self.pretrained_img_model_output_node,
            feed_dict={
                self.pretrained_img_model_input_node: resized_image
                }
        )
        return image_representation
