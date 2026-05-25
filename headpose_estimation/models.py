import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import tensorflow as tf
from tensorflow.image import resize_bilinear
from tensorflow.io import decode_jpeg
from tensorflow.layers import Dense
from tensorflow.losses import mean_squared_error
from tensorflow.python.platform import gfile

from headpose_estimation.utils.tensorflow_model_handler import TensorflowV1ModelConfig
from headpose_estimation.utils.utils import fn_from_str

logger = logging.getLogger(__name__)

ModelInput = TypeVar("ModelInput")
PredictionOutput = TypeVar("PredictionOutput")


class BaseModel(Generic[ModelInput, PredictionOutput], ABC):
    """
    Baseclass for all the models used in this project
    """

    @property
    @abstractmethod
    def prediction_ops(self) -> Dict[str, tf.Tensor]:
        pass

    @property
    @abstractmethod
    def input_placeholder(self) -> tf.Tensor:
        pass

    @abstractmethod
    def predict(
        self, session: tf.Session, prediction_input: List[ModelInput]
    ) -> List[PredictionOutput]:
        pass


class BaseAnglePredictionHeadModel(BaseModel[np.ndarray, Dict[str, float]]):
    @property
    @abstractmethod
    def training_ops(self) -> List[tf.Tensor]:
        pass

    @property
    @abstractmethod
    def ground_truth_placeholders(self) -> Dict[str, tf.Tensor]:
        pass

    @property
    @abstractmethod
    def loss_ops(self) -> Dict[str, tf.Tensor]:
        pass


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
        self._input_placeholder = tf.placeholder(
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
        session: tf.Session,
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
    def _load_graph_definition(graph_def_path: Union[str, Path]) -> tf.GraphDef:
        with gfile.FastGFile(graph_def_path, "rb") as f_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f_handle.read())
            return graph_def

    @property
    def input_placeholder(self) -> None:
        return self._input_placeholder

    @property
    def prediction_ops(self) -> Dict[str, tf.Tensor]:
        return {self.IMAGE_REPRESENTATION_KEY: self._prediction_op}

    def predict(
        self, session: tf.Session, prediction_input: List[str]
    ) -> List[np.ndarray]:

        preprocessed_image_input: List[np.ndarray] = self._preprocess_raw_images(
            session=session, input_images=prediction_input
        )

        if self.model_config.supports_batching:
            batched_preprocessed_images = np.vstack(preprocessed_image_input)
            image_representation = session.run(
                [self._image_preprocessing_output_node],
                feed_dict={
                    self.prediction_ops[
                        self.IMAGE_REPRESENTATION_KEY
                    ]: batched_preprocessed_images
                },
            )
            output_representation = np.vsplit(
                image_representation, len(image_representation)
            )

        else:
            output_representation: List[np.ndarray] = []
            for image_input in preprocessed_image_input:
                image_representation = session.run(
                    self._image_preprocessing_output_node,
                    feed_dict={
                        self.prediction_ops[self.IMAGE_REPRESENTATION_KEY]: image_input
                    },
                )
                output_representation.append(image_representation)

        return output_representation


class EulerAnglesPredictionHead(BaseAnglePredictionHeadModel):
    """
    Predicts Euler angles for yaw, pitch and roll from an image representation.

    Uses the tensorflow v1 Dense layer, this layer does not need the input dimensions it calculates them dynamically
    when the variables are initialized(with `session.run(tf.global_variables_initializer())`). At this time Tensorflow
    builds the static computation graph looks at the shape of the input tensor and uses it to create the weight matrix
    for the Dense layer.
    """

    YAW_ANGLE_KEY = "yaw"
    PITCH_ANGLE_KEY = "pitch"
    ROLL_ANGLE_KEY = "roll"

    def __init__(
        self,
        input_representation_size: int,
        layer_sizes: List[int],
        activation_function: str = "tensorflow.nn.relu",
        optimizer_function: str = "tf.train.AdamOptimizer",
    ) -> None:

        self._input_placeholder = tf.placeholder(
            tf.float32,
            shape=[None, input_representation_size],
            name="input_image_representation",
        )
        self.optimizer = optimizer_function

        activation_fn = fn_from_str(activation_function)
        yaw_angle_prediction = self.intialize_weights_for_an_angle(
            input_image_representation_tensor=self._input_placeholder,
            layer_sizes=layer_sizes,
            name="yaw_predictor",
            activation_function=activation_fn,
        )

        pitch_angle_prediction = self.intialize_weights_for_an_angle(
            input_image_representation_tensor=self._input_placeholder,
            layer_sizes=layer_sizes,
            name="pitch_predictor",
            activation_function=activation_fn,
        )

        roll_angle_prediction = self.intialize_weights_for_an_angle(
            input_image_representation_tensor=self._input_placeholder,
            layer_sizes=layer_sizes,
            name="roll_predictor",
            activation_function=activation_fn,
        )

        self._prediction_ops = {
            self.YAW_ANGLE_KEY: yaw_angle_prediction,
            self.PITCH_ANGLE_KEY: pitch_angle_prediction,
            self.ROLL_ANGLE_KEY: roll_angle_prediction,
        }

    def intialize_weights_for_an_angle(
        self,
        input_image_representation_tensor: tf.Tensor,
        layer_sizes: List[int],
        name: str,
        activation_function: Callable[[tf.Tensor], tf.Tensor],
    ) -> tf.Tensor:
        previous_layer_output = input_image_representation_tensor
        with tf.name_scope(name):
            for _, output_size in enumerate(layer_sizes):
                layer = Dense(
                    units=output_size,
                    use_bias=False,
                    activation=activation_function,
                    trainable=True,
                )
                previous_layer_output = layer(previous_layer_output)

        return previous_layer_output

    def initialize_training_ops(
        self, prediction_tensors: Dict[str, tf.Tensor]
    ) -> List[tf.Tensor]:
        yaw_ground_truth_placeholder = tf.placeholder(
            tf.float32, shape=[None], name="input_yaw_gt"
        )
        roll_ground_truth_placeholder = tf.placeholder(
            tf.float32, shape=[None], name="input_roll_gt"
        )
        pitch_ground_truth_placeholder = tf.placeholder(
            tf.float32, shape=[None], name="input_pitch_gt"
        )

        self._ground_truth_placeholder = {
            self.YAW_ANGLE_KEY: yaw_ground_truth_placeholder,
            self.ROLL_ANGLE_KEY: roll_ground_truth_placeholder,
            self.PITCH_ANGLE_KEY: pitch_ground_truth_placeholder,
        }

        yaw_loss = mean_squared_error(
            yaw_ground_truth_placeholder, tf.squeeze(self.yaw_angle_prediction, axis=1)
        )
        roll_loss = mean_squared_error(
            roll_ground_truth_placeholder,
            tf.squeeze(self.roll_angle_prediction, axis=1),
        )
        pitch_loss = mean_squared_error(
            pitch_ground_truth_placeholder,
            tf.squeeze(self.pitch_angle_prediction, axis=1),
        )

        total_loss = yaw_loss + roll_loss + pitch_loss

        optimizer_node = self.optimizer(total_loss)

        self._loss_ops = {
            "total": total_loss,
            self.YAW_ANGLE_KEY: yaw_loss,
            self.ROLL_ANGLE_KEY: roll_loss,
            self.PITCH_ANGLE_KEY: pitch_loss,
        }

        self._training_ops = {
            "optimizer_node": optimizer_node,
        }

    @property
    def prediction_ops(self) -> Dict[str, tf.Tensor]:
        return self._prediction_ops

    @property
    def input_placeholder(self) -> tf.Tensor:
        return self._input_placeholder

    @property
    def ground_truth_placeholders(self) -> Dict[str, tf.Tensor]:
        return self._ground_truth_placeholder

    @property
    def training_ops(self) -> Dict[str, tf.Tensor]:
        return self._training_ops

    @property
    def loss_ops(self) -> Dict[str, tf.Tensor]:
        return self._loss_ops

    def train(
        self,
        session: tf.Session,
        input: List[np.ndarray],
        ground_truths: Dict[Literal["yaw", "pitch", "roll"], List[float]],
    ) -> List[float]:

        _, total_loss, yaw_loss, roll_loss, pitch_loss = session.run(
            [*list(self.training_ops.values()), *list(self.loss_ops.values())],
            feed_dict={
                self.input_placeholder: input,
                self.ground_truth_placeholders[self.YAW_ANGLE_KEY]: ground_truths[
                    self.YAW_ANGLE_KEY
                ],
                self.ground_truth_placeholders[self.PITCH_ANGLE_KEY]: ground_truths[
                    self.PITCH_ANGLE_KEY
                ],
                self.ground_truth_placeholders[self.ROLL_ANGLE_KEY]: ground_truths[
                    self.ROLL_ANGLE_KEY
                ],
            },
        )

        return total_loss, yaw_loss, roll_loss, pitch_loss

    def predict(
        self, session: tf.Session, prediction_input: List[np.ndarray]
    ) -> List[Dict[str, float]]:
        """Predict the Euler Angles using the image representations

        Args:
            image_input_representation (np.ndarray): Image representations of the shape
                                                     [Batch_size, 1, image_representation_size]

        Returns:
            np.ndarray: Predicted Euler angles(in the order of yaw, pitch, roll) of the shape [Batch_size, 3]
        """
        stacked_prediction_input = np.vstack(prediction_input)
        yaw_angle, pitch_angle, roll_angle = session.run(
            list(self.prediction_ops.values()),
            feed_dict={self.input_placeholder: stacked_prediction_input},
        )

        output: List[Dict[str, float]] = []
        for i in len(prediction_input):
            output.append(
                {
                    self.YAW_ANGLE_KEY: yaw_angle[i][0],
                    self.ROLL_ANGLE_KEY: roll_angle[i][0],
                    self.PITCH_ANGLE_KEY: pitch_angle[i][0],
                }
            )

        return output


class EulerAnglesPredictionModel(BaseModel[str, Dict[str, float]]):
    """
    Container class for a base pretrained image backbone model and a prediction head.
    Allows providing image data and getting the angle predictions directly from it.
    """

    def __init__(
        self,
        backbone_image_model: PretrainedBackBoneImageModel,
        euler_angle_prediction_head: EulerAnglesPredictionHead,
    ) -> None:

        self._backbone_image_model = backbone_image_model
        self._angle_prediction_head = euler_angle_prediction_head

    @property
    def input_placeholder(self) -> tf.Tensor:
        return self._backbone_image_model.input_placeholder

    @property
    def prediction_ops(self) -> Dict[str, tf.Tensor]:
        return self._angle_predictin_head.prediction_ops

    def predict(
        self, session: tf.Session, prediction_input: List[str]
    ) -> List[Dict[str, float]]:
        backbone_model_representations: List[np.ndarray] = (
            self._backbone_image_model.predict(
                session=session, prediction_input=prediction_input
            )
        )

        output_angles = self.angle_prediction_head.predict(
            session=session, prediction_input=backbone_model_representations
        )

        return output_angles
