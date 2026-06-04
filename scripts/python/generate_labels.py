import argparse
import csv
import logging
import os
import random
from pathlib import Path
from typing import Dict, List

from scipy.io import loadmat

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-i",
    "--input-path",
    type=str,
    help="Path to the folder containing the images and the ground truth euler angle values",
  )
  parser.add_argument(
    "-o",
    "--output-path",
    type=str,
    help="Path to the directory where the processed dataset will be stored",
  )
  parser.add_argument(
    "-n",
    "--sub-sample",
    type=int,
    default=0,
    help="Number of examples to subsample from the dataset if set to 0, no subsampling is done",
  )

  return parser.parse_args()


def main() -> None:

  img_filepath_identifier = "image_filepath"
  ground_truth_filepath_identifier = "ground_truth_filepath"
  args = parse_args()

  image_name_to_data_paths: Dict[str, List[str]] = dict()
  img_filepaths = list(Path(args.input_path).rglob("*.jpg"))

  for path in img_filepaths:
    file_name = path.name
    image_name_to_data_paths[file_name] = {
      img_filepath_identifier: path,
      ground_truth_filepath_identifier: f"{str(path).split('.jpg')[0]}.mat",
    }

  shuffled_data_keys = list(image_name_to_data_paths.keys())
  random.shuffle(shuffled_data_keys)

  if args.sub_sample != 0:
    shuffled_data_keys = shuffled_data_keys[: args.sub_sample]

  ground_truths = []
  for key in shuffled_data_keys:
    img_data = loadmat(image_name_to_data_paths[key][ground_truth_filepath_identifier])
    try:
      pitch, yaw, roll = img_data["Pose_Para"][0, 0:3]
      yaw = -yaw

      ground_truths.append(
        (
          key,
          image_name_to_data_paths[key][img_filepath_identifier],
          pitch,
          yaw,
          roll,
        )
      )
    except KeyError as e:
      logger.warning(f"Did not find key Pose_Para in {image_name_to_data_paths[key][img_filepath_identifier]}")

  ground_truths.insert(0, ("img_name", "img_path", "pitch", "yaw", "roll"))

  if not os.path.exists(Path(args.output_path)):
    os.makedirs(args.output_path)

  with open(Path(args.output_path).joinpath("train_dataset.csv"), "w") as ground_truth_file:
    tsvwriter = csv.writer(ground_truth_file, delimiter="\t")
    tsvwriter.writerows(ground_truths)


if __name__ == "__main__":
  main()
