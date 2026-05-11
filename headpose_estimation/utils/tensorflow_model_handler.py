from __future__ import annotations
from dataclasses import dataclass
import os
from functools import partial
import logging
from pathlib import Path
import tarfile
from typing import List, Tuple, Dict, Union
from urllib import request
import yaml


logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TensorflowV1ModelConfig:
    """
    Stores the information about the Tensorflow V1 models.
    """

    architecture: str
    graph_definition_path: str
    download_url: str
    input_node_name: str
    output_node_name: str
    output_tensor_size: int
    image_input_size: Tuple[int, int]
    image_depth: int
    batch_size: int = 1

    @classmethod
    def from_dict(cls, input_config: Dict[str, Union[str, int]]) -> TensorflowV1ModelConfig:
        return cls(**input_config)


class TensorflowV1ModelHandler:
    """
    Handler for TF1 models. Given a model name it performs the following functions:
    - Download the model if not present on disk
    - Provide information about the model, for more information check out the `TensorflowV1ModelConfig` class.
    """

    def __init__(
        self,
        tf_models_config_path: str
    ):
        with open(tf_models_config_path) as f:
            models_list: List[Dict[str, Union[str, int, Tuple[int, int]]]] = yaml.safe_load(f)

        self.models = {
            model["architecture"]: TensorflowV1ModelConfig.from_dict(model) for model in models_list
        }

    def get_model_info(self, architecture: str) -> TensorflowV1ModelConfig:
        if architecture not in self.models:
            raise ValueError(
                f"{architecture} not specified in the tf model configs, check the architecture name or update the model configs"
            )
        return self.models[architecture]

    def get_model(self, architecture: str, output_path: str) -> Path:
        model_folder = Path(output_path)
        model_path = model_folder / architecture
        if model_path.exists():
            return model_path

        logger.info("Downloading model %s", architecture)
        downloaded_model_path = self._download_model(architecture, model_folder)
        self._extract_model_file(downloaded_model_path, model_folder)

        return model_path

    @staticmethod
    def __model_download_progress_bar(count: int, block_size: float, total_size: float, filename: str) -> None:
        logger.info("Downloaded model %s %0.2f%%", filename, float((count*block_size*100)/total_size))

    def _download_model(self, architecture: str, output_path: Path) -> Path:
        download_url = self.models[architecture].download_url
        model_tarfile = download_url.split("/")[-1]
        downloaded_model_filepath = output_path / model_tarfile

        request.urlretrieve(
            url=download_url,
            filename=downloaded_model_filepath,
            reporthook=partial(self.__model_download_progress_bar, filename=architecture)
        )

        return downloaded_model_filepath

    @staticmethod
    def _extract_model_file(model_path: Path, unzip_dir: Path) -> None:
        logger.info("Extracting %s into %s", model_path, unzip_dir)

        with tarfile.open(model_path, "r:gz") as tar_file:
            tar_file.extractall(unzip_dir, filter="tar")
        
        os.remove(model_path)
