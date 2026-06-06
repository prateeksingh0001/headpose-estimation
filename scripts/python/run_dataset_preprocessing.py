import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Union

from headpose_estimation.dataset_preprocessors import ProcessedFrameType, ThreeHudredWLPPreprocessor, UPNAPreprocessor
from headpose_estimation.schema import DataPreprocessingConfig
from headpose_estimation.utils.utils import cls_from_str

DataPreprocessorType = Union[UPNAPreprocessor, ThreeHudredWLPPreprocessor]

# Datapreprocessing constant
IMAGE_FOLDER_NAME = "images"
GROUND_TRUTH_FILE_NAME = "ground_truth.json"


def main() -> None:
  args = parse_args()

  preprocessing_config: DataPreprocessingConfig = DataPreprocessingConfig.load_from_config(
    config_path=args["config_path"]
  )

  if not preprocessing_config.output_path.exists():
    preprocessing_config.output_path.mkdir(exist_ok=True)

  image_folder_path = preprocessing_config.output_path / IMAGE_FOLDER_NAME
  if not image_folder_path.exists():
    image_folder_path.mkdir(exist_ok=True)

  output_frame_information: ProcessedFrameType = []
  for dataset in preprocessing_config.datasets:
    dataset_preprocessor: DataPreprocessorType = cls_from_str(dataset.preprocessor_class)(
      dataset_name=dataset.dataset_name,
      dataset_path=preprocessing_config.dataset_root_path / dataset.dataset_name,
      output_path=image_folder_path,
    )
    output_frame_information.extend(dataset_preprocessor.process_dataset(sample_size=dataset.subsample_size))

  with open(preprocessing_config.output_path / GROUND_TRUTH_FILE_NAME, "w") as f:
    json.dump(output_frame_information, f)


def parse_args() -> Dict[str, Any]:
  parser = ArgumentParser()
  parser.add_argument(
    "-c", "--config", type=str, required=True, dest="config_path", help="Path to the data preprocessing config"
  )
  return vars(parser.parse_args())


if __name__ == "__main__":
  main()
