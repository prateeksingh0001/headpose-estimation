from functools import partial
import logging
from pathlib import Path
import tarfile
from typing import List, Tuple, Dict, Union
from urllib import request
import yaml
from yaml import Loader

from headpose_estimation.utils.pretrained_model_config import ModelConfig

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


class TensorflowModelHandler:
    def __init__(
        self,
        tf_models_config_path: str
    ):
        models_list: List[Dict[str, Union[str, int, Tuple[int, int]]]] = yaml.load(open(tf_models_config_path), Loader=Loader)
        self.models = {
            model["architecture"]: ModelConfig.from_dict(model) for model in models_list
        }

    def get_model_info(self, architecture: str) -> Dict[str, Union[str, int, Tuple[int, int]]]:
        if architecture not in self.models:
            raise ValueError(f"{architecture} not specfied in the tf model configs, check the architecture name or update the model configs")

        return self.models.get(architecture).to_dict()

    @staticmethod
    def __model_download_progressbar(count: int, block_size: float, total_size: float, filename: str) -> None:
        logger.info("Downloaded model %s %0.2f%%"%(filename, float((count*block_size*100)/total_size)))

    def _download_model(self, architecture: str, output_path: str) -> str:
        download_url = self.models.get(architecture).download_url
        model_tarfile = download_url.split("/")[-1]
        downloaded_model_filepath = Path(output_path).joinpath(model_tarfile)

        request.urlretrieve(
            url=download_url,
            filename=downloaded_model_filepath,
            reporthook=partial(self.__model_download_progressbar, filename=architecture)
        )
        logger.info("\n")

        return downloaded_model_filepath

    @staticmethod
    def _extract_model_file(model_path: str, unzip_dir: str) -> str:
        logger.info("Extracting %s into %s"%(model_path, unzip_dir))
        tarfile.open(model_path, "r:gz").extractall(unzip_dir)

        return unzip_dir

    def get_model(self, architecture: str, output_path: str) -> str:
        logger.info("Donwloading model %s"%(architecture))
        downloaded_model_path = self._download_model(architecture, output_path)
        model_dir = self._extract_model_file(downloaded_model_path, output_path)

        return output_path
