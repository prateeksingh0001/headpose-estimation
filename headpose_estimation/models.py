import logging
from abc import ABC
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import expand_dims
from tensorflow.image import resize_bilinear
from tensorflow.io import decode_jpeg
from tensorflow.layers import Dense
from tensorflow.python.platform import gfile

from headpose_estimation.utils.tensorflow_model_handler import TensorflowV1ModelConfig
from headpose_estimation.utils.utils import fn_from_str

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Baseclass for all the models used in this project
    """

    def __init__(self, session: tf.Session, **kwargs):
        self.session = session


class PretrainedBackBoneImageModel(BaseModel):
    """
    Handles the pretrained backbone models.
    """

    IMAGE_NORMALIZATION_CENTER_CONTANT = 128
    IMAGE_NORMALIZATION_MEAN_MULTIPLIER = 0.0078125

    def __init__(
        self, tf_model_config: TensorflowV1ModelConfig, session: tf.Session = None
    ) -> None:

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
    def _load(graph_def_path: Union[str, Path]) -> tf.GraphDef:
        with gfile.FastGFile(graph_def_path, "rb") as f_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f_handle.read())
            return graph_def

    def forward(self, image_data: List[str]) -> List[np.ndarray]:
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
                output_representation.append(image_representation)

        return output_representation


class EulerAnglesPredictionHead(BaseModel):
    """
    Predicts Euler angles for yaw, pitch and roll from an image representation.

    Uses the tensorflow v1 Dense layer, this layer does not need the input dimensions it calculates them dynamically
    when the variables are initialized(with `session.run(tf.global_variables_initializer())`). At this time Tensorflow
    builds the static computation graph looks at the shape of the input tensor and uses it to create the weight matrix
    for the Dense layer.
    """

    def __init__(
        self,
        input_image_representation_tensor: tf.Tensor,
        layer_sizes: List[int],
        session: tf.Session,
        activation_function: str = "tensorflow.nn.relu",
    ) -> None:

        self.input_image_representation_tensor = input_image_representation_tensor
        activation_fn = fn_from_str(activation_function)

        self.yaw_angle_prediction = self.intialize_weights_for_angle(
            input_image_representation_tensor=self.input_image_representation_tensor,
            layer_sizes=layer_sizes,
            name="yaw_predictor",
            activation_function=activation_fn,
        )

        self.pitch_angle_prediction = self.intialize_weights_for_angle(
            input_image_representation_tensor=self.input_image_representation_tensor,
            layer_sizes=layer_sizes,
            name="pitch_predictor",
            activation_function=activation_fn,
        )

        self.roll_angle_prediction = self.intialize_weights_for_angle(
            input_image_representation_tensor=self.input_image_representation_tensor,
            layer_sizes=layer_sizes,
            name="roll_predictor",
            activation_function=activation_fn,
        )
        super().__init__(session=session)

    def intialize_weights_for_angle(
        self,
        input_image_representation_tensor: tf.Tensor,
        layer_sizes: List[int],
        name: str,
        activation_function: Callable[[tf.Tensor], tf.Tensor],
    ) -> tf.Tensor:
        previous_layer_output = input_image_representation_tensor
        with tf.name_scope(name):
            for i, output_size in enumerate(layer_sizes):
                layer = Dense(
                    units=output_size,
                    use_bias=False,
                    activation=activation_function,
                    trainable=True,
                )
                previous_layer_output = layer(previous_layer_output)

        return previous_layer_output

    def forward(self, image_input_representation: np.ndarray) -> np.ndarray:
        """Predict the Euler Angles using the image representations

        Args:
            image_input_representation (np.ndarray): Image representations of the shape
                                                     [Batch_size, 1, image_representation_size]

        Returns:
            np.ndarray: Predicted Euler angles(in the order of yaw, pitch, roll) of the shape [Batch_size, 3]
        """
        yaw_angle, pitch_angle, roll_angle = self.session.run(
            [
                self.yaw_angle_prediction,
                self.pitch_angle_prediction,
                self.roll_angle_prediction,
            ],
            feed_dict={
                self.input_image_representation_tensor: image_input_representation
            },
        )
        print(yaw_angle.shape)
        print(pitch_angle.shape)
        print(roll_angle.shape)
        batched_angles = np.hstack((yaw_angle, pitch_angle, roll_angle))

        return batched_angles


class EulerAnglesPredictionModel(BaseModel):
    def __init__(
        self,
        backbone_image_model: PretrainedBackBoneImageModel,
        session: tf.Session,
        euler_angle_prediction_head: Optional[EulerAnglesPredictionHead] = None,
        **kwargs,
    ) -> None:

        self.backbone_image_model = backbone_image_model
        self.angle_prediction_head = euler_angle_prediction_head

        if not euler_angle_prediction_head:
            euler_angle_prediction_head = EulerAnglesPredictionHead(
                input_image_representation_tensor=tf.placeholder(
                    tf.float32,
                    shape=[
                        None,
                        self.backbone_image_model.model_config.output_tensor_size,
                    ],
                    name="angle_prediction_input",
                ),
                layer_sizes=[512, 1],
            )
        self.euler_angle_prediction_head = euler_angle_prediction_head
        super().__init__(session=session)

    def get_backbone_image_representation(
        self, image_data: List[str]
    ) -> List[np.ndarray]:
        return self.backbone_image_model.forward(image_data=image_data)

    def get_angle_predictions_from_img_representation(
        self, image_representations: np.ndarray
    ):
        return self.angle_prediction_head.forward(
            image_input_representation=image_representations
        )

    def forward(self, image_data: List[str]) -> np.ndarray:
        backbone_model_representation = self.backbone_image_model.forward(
            image_data=image_data
        )

        batched_backbone_model_representations = np.vstack(
            backbone_model_representation
        )
        output_angles = self.angle_prediction_head.forward(
            image_input_representation=batched_backbone_model_representations
        )
        return output_angles
