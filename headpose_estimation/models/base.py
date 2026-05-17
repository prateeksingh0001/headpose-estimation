import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import expand_dims
from tensorflow.image import resize_bilinear
from tensorflow.io import decode_jpeg
from tensorflow.python.platform import gfile

from headpose_estimation.utils.tensorflow_model_handler import TensorflowV1ModelConfig

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Baseclass for all the models used in this project
    """

    def __init__(self, session: tf.Session, **kwargs):
        self.session = session


class PretrainedBackBoneImageModel(BaseModel):
    """
    Baseclass for all the pretrained backbone image models.
    """

    IMAGE_NORMALIZATION_CENTER_CONTANT = 128
    IMAGE_NORMALIZATION_MEAN_MULTIPLIER = 0.0078125

    def __init__(
        self, tf_model_config: TensorflowV1ModelConfig, session: tf.Session = None
    ):

        self.model_config = tf_model_config
        self.image_preprocessing_input_node = tf.placeholder(
            tf.string, shape=[], name="image_bytes"
        )
        self.image_preprocessing_output_node = self._create_preprocessing_graph(
            raw_image_data_tensor=self.image_preprocessing_input_node
        )

        self.pretrained_img_model_input_node, self.pretrained_img_model_output_node = (
            self._create_model_graph(
                graph_def_path=tf_model_config.graph_definition_path,
                input_node_name=tf_model_config.input_node_name,
                output_node_name=tf_model_config.output_node_name,
            )
        )

        self.session = session

    def _create_preprocessing_graph(
        self, raw_image_data_tensor: tf.Tensor
    ) -> tf.Tensor:
        image_data = expand_dims(decode_jpeg(raw_image_data_tensor, channels=3), axis=0)
        resized_image = resize_bilinear(
            images=image_data, size=self.model_config.image_input_size
        )
        scaled_image_data = (
            resized_image - self.IMAGE_NORMALIZATION_CENTER_CONTANT
        ) * self.IMAGE_NORMALIZATION_MEAN_MULTIPLIER
        return scaled_image_data

    def _preprocess_raw_images(
        self,
        input_images: List[str],
        *args,
        **kwargs,
    ) -> List[np.ndarray]:
        """
        Given a list of images as bytes encoded as string, it decodes them as JPEG, load the content of the image
        resizes them to models dimensions and normalizes the pixel values to be between -1.0 to 1.0.
        Model dimensions are fetched from self.model_config.image_input_size

        Args:
            input_images (List[str]): Input image passed as list of bytes encoded as a string

        Returns:
            List[np.ndarray]: List of resized images, where, each image is of the dimension
                              (1, self.model_config.image_input_size[0], self.model_config.image_input_size[1], 3)
        """

        preprocessed_images = list(
            map(
                lambda raw_image_data: self.session.run(
                    self.image_preprocessing_output_node,
                    feed_dict={self.image_preprocessing_input_node: raw_image_data},
                ),
                input_images,
            )
        )

        return preprocessed_images

    def _create_model_graph(
        self, graph_def_path: str, input_node_name: str, output_node_name: str
    ) -> Tuple[Union[tf.Operation, tf.Tensor], Union[tf.Operation, tf.Tensor]]:

        graph_def = self._load(graph_def_path)

        input_tensor, output_tensor = tf.import_graph_def(
            graph_def, return_elements=[input_node_name, output_node_name]
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

    def forward(self, image_data: List[bytes]) -> List[tf.Tensor]:
        preprocessed_image_input: List[np.ndarray] = self._preprocess_raw_images(
            input_images=image_data
        )

        if self.model_config.supports_batching:
            batched_preprocessed_images = np.vstack(preprocessed_image_input, axis=0)
            image_representation = self.session.run(
                self.pretrained_img_model_output_node,
                feed_dict={
                    self.pretrained_img_model_input_node: batched_preprocessed_images
                },
            )

            output_representation = np.vsplit(image_representation, len(image_data))
        else:
            output_representation: List[np.ndarray] = []
            for image_input in preprocessed_image_input:
                image_representation = self.session.run(
                    self.pretrained_img_model_output_node,
                    feed_dict={self.pretrained_img_model_input_node: image_input},
                )

        return output_representation
