from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.image import resize_bilinear
from tensorflow.io import decode_jpeg
from tensorflow.python.platform import gfile

from headpose_estimation.models.base import BaseModel
from headpose_estimation.utils.tensorflow_model_handler import TensorflowV1ModelConfig


class PretrainedBackBoneImageModel(BaseModel[str, np.ndarray]):
    """
    Handles the pretrained backbone models.
    """

    PIXEL_NORMALIZATION_MEAN = 128
    PIXEL_NORMALIZATION_MULTIPLIER = 0.0078125
    IMAGE_REPRESENTATION_KEY = "image_representation"

    def __init__(self, tf_model_config: TensorflowV1ModelConfig) -> None:
        """
        Loads a pretrained image model from it graph definition and processes images to generate representations
        using the model

        Args:
            tf_model_config (TensorflowV1ModelConfig): Config for the tensorflow model.
        """

        self.model_config = tf_model_config
        self._input_placeholder = tf.compat.v1.placeholder(
            tf.string, shape=[], name="input_image_bytes"
        )
        self._image_preprocessing_output_node = self._create_preprocessing_graph(
            input_raw_image_tensor=self._input_placeholder
        )

        self._pretrained_img_model_input_node, self._prediction_op = (
            self._create_model_graph(
                graph_def_path=tf_model_config.graph_definition_path,
                input_node_name=tf_model_config.input_node_name,
                output_node_name=tf_model_config.output_node_name,
            )
        )

    def _create_preprocessing_graph(
        self, input_raw_image_tensor: tf.Tensor
    ) -> tf.Tensor:
        """Sets up the tensorflow nodes for decoding, resizing and scaling the images

        Args:
            input_raw_image_tensor (tf.Tensor): Tensorflow tensor/placeholder which will contain the raw image data
                                                (of the type str) during runtime.

        Returns:
            tf.Tensor: Output tensor containing the processed image of the size (1, height, width, num_channels)
        """

        image_data = tf.expand_dims(
            decode_jpeg(input_raw_image_tensor, channels=3), axis=0
        )
        resized_image = resize_bilinear(
            images=image_data, size=self.model_config.image_input_size
        )
        scaled_image_data = (
            resized_image - self.PIXEL_NORMALIZATION_MEAN
        ) * self.PIXEL_NORMALIZATION_MULTIPLIER
        return scaled_image_data

    def _preprocess_raw_images(
        self,
        session: tf.compat.v1.Session,
        input_images: List[str],
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
                lambda raw_image_data: session.run(
                    self._image_preprocessing_output_node,
                    feed_dict={self._input_placeholder: raw_image_data},
                ),
                input_images,
            )
        )

        return preprocessed_images

    def _create_model_graph(
        self, graph_def_path: str, input_node_name: str, output_node_name: str
    ) -> Tuple[Union[tf.Operation, tf.Tensor], Union[tf.Operation, tf.Tensor]]:
        """Reads and imports the graph definition of a tensorflow model from disk

        Args:
            graph_def_path (str): Path to the tensorflow graph definition on disk
            input_node_name (str): Name of the input node in the tensorflow graph
            output_node_name (str): Name of the output node in the tensorflow graph

        Returns:
            Tuple[Union[tf.Operation, tf.Tensor], Union[tf.Operation, tf.Tensor]]: Input and output tensors in the graph
        """

        graph_def = self._load_graph_definition(graph_def_path)

        input_tensor, output_tensor = tf.import_graph_def(
            graph_def, return_elements=[input_node_name, output_node_name]
        )

        return input_tensor, output_tensor

    @staticmethod
    def _load_graph_definition(
        graph_def_path: Union[str, Path],
    ) -> tf.compat.v1.GraphDef:
        with gfile.FastGFile(graph_def_path, "rb") as f_handle:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f_handle.read())
            return graph_def

    @property
    def input_placeholder(self) -> None:
        return self._input_placeholder

    @property
    def prediction_ops(self) -> Dict[str, tf.Tensor]:
        return {self.IMAGE_REPRESENTATION_KEY: self._prediction_op}

    def predict(
        self, session: tf.compat.v1.Session, prediction_input: List[str]
    ) -> List[np.ndarray]:

        preprocessed_image_input: List[np.ndarray] = self._preprocess_raw_images(
            session=session, input_images=prediction_input
        )

        if self.model_config.supports_batching:
            batched_preprocessed_images = np.vstack(preprocessed_image_input)
            image_representation = session.run(
                self.prediction_ops[self.IMAGE_REPRESENTATION_KEY],
                feed_dict={
                    self._pretrained_img_model_input_node: batched_preprocessed_images
                },
            )
            output_representation = np.vsplit(
                image_representation, len(image_representation)
            )

        else:
            output_representation: List[np.ndarray] = []
            for image_input in preprocessed_image_input:
                image_representation = session.run(
                    self.prediction_ops[self.IMAGE_REPRESENTATION_KEY],
                    feed_dict={self._pretrained_img_model_input_node: image_input},
                )
                output_representation.append(image_representation)

        return output_representation
