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


class PretrainedImageModel(BaseModel):

    def __init__(
        self,
        architecture: str,
        model_path: str
    ):architecture, 

        self.filename = None
        self.filepath = None
        self.flags = None

        self.model_info = self.create_model_info(architecture)
        self.maybe_download_and_extract(model_path)

    def maybe_download_and_extract(self, model_path: str) -> None:
        if os.path.exists(model_path):
            logger.INFO("Not extracting or downloading files, model already present in disk")
        else:
            self.create_dest_dir(model_path)
            self.download(model_path)
            self.extract()

    def download(self, model_path: str) -> None:
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                "\r>> Downloading %s %.1f%%"
                % (self.filename, float(count * block_size) / float(total_size) * 100.0)
            )
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(self.model_info["data_url"], model_path, _progress)  #### filepath clobbered !!!
        
        statinfo = os.stat(filepath)
        logger.info("Successfully downloaded %s %d bytes.", self.filename, statinfo.st_size)

    def extract(self) -> None:
        logger.info(f"Extracting file from {self.filepath}")
        tarfile.open(self.filepath, "r:gz").extractall(self.flags.model_dir)

    @staticmethod
    def create_dest_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

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
            tf.logging.error("Couldn't understand architecture name '%s'", architecture)
            raise ValueError("Unknown architecture", architecture)

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

    def create_model_graph(self):
        if not self.model_info:
            tf.logging.error("Did not recognize architecture flag")
            return -1

        with tf.Graph().as_default() as graph:
            model_path = os.path.join(
                FLAGS.model_dir, self.model_info["model_file_name"]
            )
            logger.info(f"Model path: {model_path}")
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
