from typing import Dict, Any

import logging
import os
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
from tensorflow.python.platform import gfile

from headpose_estimation.models.base_model import BaseModel


logger = logging.getLogger(__name__)


class PretrainedBaseImageModel(BaseModel):
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

    def __init__(self, architecture: str, model_path: str):
        """Initializes the instance based on model architecture and the model
        save path

        Args:
            architecture (str): name of the vision model architecture
            model_path (str): model path on disk where the model is saved or will
                              be downloaded to.
        """

        self.filename = None

        self.model_info = self.create_model_info(architecture)
        if not os.path.exists(model_path):
            self.download_and_extract(model_path)
        else:
            logger.info("model present on disk at %s", model_path)

        self.model_graph, self.bottleneck_tensor, self.resized_input_tensor = self.create_model_graph(model_path)

    def download_and_extract(self, model_path: str) -> None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        downloaded_model_artifact_path = self.download(model_path)
        self.extract(downloaded_model_artifact_path, model_path)

    def download(self, model_path: str) -> str:
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                "\r>> Downloading %s %.1f%%"
                % (self.filename, float(count * block_size) / float(total_size) * 100.0)
            )
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(
            self.model_info["data_url"], model_path, _progress
        )
        statinfo = os.stat(filepath)

        logger.info(
            "Successfully downloaded %s %d bytes.", self.filename, statinfo.st_size
        )
        return filepath

    @staticmethod
    def extract(downloaded_model_zip_path: str, unzip_dir: str) -> None:
        logger.info("Extracting file from %s", downloaded_model_zip_path)
        tarfile.open(downloaded_model_zip_path, "r:gz").extractall(unzip_dir)

    def create_model_info(self, architecture: str) -> Dict[str, Any]:
        architecture = architecture.lower()
        is_quantized = False
        if architecture == "inception_v3":
            model_file_name = "classify_image_graph_def.pb"
            data_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"
            resized_input_tensor_name = "Mul:0"
            bottleneck_tensor_name = "pool_3/_reshape:0"
            bottleneck_tensor_size = 2048
            input_width = 299
            input_height = 299
            input_depth = 3
            input_mean = 128
            input_std = 128

        elif architecture.startswith("mobilenet_"):
            parts = architecture.split("_")
            if len(parts) != 3 and len(parts) != 4:
                tf.logging.error(
                    "Couldn't understand architecture name '%s'", architecture
                )
                return

            version_string = parts[1]
            if version_string not in ["1.0", "0.75", "0.5", "0.25"]:
                tf.logging.error(
                    """The Mobilenet version should be '1.0', '0.75', '0.5', or '0.25', but found '%s' for
                     architecture '%s'""",
                    version_string,
                    architecture,
                )
                return

            size_string = parts[2]
            if size_string not in ["224", "192", "160", "128"]:
                tf.logging.error(
                    """The Mobilenet input size should be '224', '192', '160', or '128', but found '%s'
                     for architecture '%s'""",
                    size_string,
                    architecture,
                )
                return

            is_quantized = len(parts) != 3
            if not is_quantized:
                if parts[3] != "quant":
                    tf.logging.error(
                        "Couldn't understand architecture suffix '%s' for '%s'",
                        parts[3],
                        architecture,
                    )
                    return

            data_url = "http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/"
            model_name = "mobilenet_v1_" + version_string + "_" + size_string

            if is_quantized:
                model_name += "_quant"

                model_file_name = model_name + "_frozen.pb"
                data_url += model_name + ".tgz"
                resized_input_tensor_name = "input:0"
                bottleneck_tensor_name = "MobilenetV1/Predictions/Reshape:0"
                bottleneck_tensor_size = 1001
                input_width = int(size_string)
                input_height = int(size_string)
                input_depth = 3
                input_mean = 127.5
                input_std = 127.5
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        return {
            "data_url": data_url,
            "bottleneck_tensor_name": bottleneck_tensor_name,
            "bottleneck_tensor_size": bottleneck_tensor_size,
            "input_width": input_width,
            "input_height": input_height,
            "resized_input_tensor_name": resized_input_tensor_name,
            "input_depth": input_depth,
            "model_file_name": model_file_name,
            "input_mean": input_mean,
            "input_std": input_std,
            "quantize_layer": is_quantized,
        }

    def create_model_graph(self, model_dir: str):
        if not self.model_info:
            tf.logging.error("Did not recognize architecture flag")
            return -1

        with tf.Graph().as_default() as graph:
            model_path = os.path.join(model_dir, self.model_info["model_file_name"])
            logger.info("Model path: %s", model_path)
            with gfile.FastGFile(model_path, "rb") as f_handle:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f_handle.read())
                bottleneck_tensor, resized_input_tensor = tf.import_graph_def(
                    graph_def,
                    name="",
                    return_elements=[
                        self.model_info["bottleneck_tensor_name"],
                        self.model_info["resized_input_tensor_name"],
                    ],
                )

        return graph, bottleneck_tensor, resized_input_tensor

    def forward(self, *arg, **kwargs) -> tf.Tensor:
        raise NotImplementedError(f"forward method is not implemented for {self.__class__.__name__}")
