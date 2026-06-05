"""
Preprocessors for any dataset required for finetuning the angle prediction heads.
Steps:
- Read the dataset specific directory structure
- Find the path to the images and the Euler angles
- Create a JSON file containing a list of dicts with the following fields
    - id -> dataset_name_index
    - image_path
    - yaw_ground_truth -> yaw angle
    - pitch_ground_truth -> pitch angle
    - roll_ground_truth -> roll angle
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class UPNAPreprocessor:
  """
  The UPNA dataset consists of mp4 videos each of 300 frames and the ground truth values, in degrees, for each frame.
  the ground truth files(user_<n>_video_<k>_groundtruth3D.txt) files contain 6 columns contain the
  translation(in milimeters) in the first three columns and the roll, yaw and pitch values in degrees in the last
  three columns:
  Tx, Ty, Tz, Roll, Yaw, Pitch


  The dataset consists of two ground truth files `user_n_video_n_groundtruth3D.txt` and
  `user_n_video_n_groundtruth3D_zeroed.txt` files, the zeroed files contain the relative difference between the
  headpose of a particular video frame from the first frame of the video. We on the other hand have to predict the
  absolute headpose angles so for our case we'll use the non-zeroed ground truth file which contains the absolute
  headpose values for every frame.

  This preprocessor class will take each mp4 video and store the individual frames as jpeg images on the disk and
  create a JSON file containing the path to each image and its ground truth.
  """

  GROUND_TRUTH_FILE_HEADERS = [
    "translation_x",
    "translation_y",
    "translation_z",
    "roll",
    "yaw",
    "pitch",
  ]

  def __init__(self, dataset_name: str, dataset_path: str, output_path: str):
    self._dataset_name = dataset_name
    self._dataset_path = Path(dataset_path)
    self._output_path = Path(output_path)
    self._images_path = Path(output_path)

  def process_single_video(
    self,
    video_path: Path,
    ground_truth: pd.DataFrame,
    running_sample_size: int,
    total_sample_size: Optional[int] = None,
  ) -> List[Dict[str, Union[str, float]]]:

    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
      raise RuntimeError(f"Could not open the file {video_path}")

    frame_index = 0
    processed_frames = []
    while True:
      ret, frame = video_capture.read()

      if not ret:
        break

      frame_name = f"{self._dataset_name}_{video_path.stem}_{frame_index}"
      frame_image_path = self._images_path / f"{frame_name}.jpeg"
      success = cv2.imwrite(frame_image_path, frame)

      processed_frames.append(
        {
          "id": frame_name,
          "image_path": str(frame_image_path),
          "yaw_ground_truth": ground_truth.iloc[frame_index]["yaw"],
          "pitch_ground_truth": ground_truth.iloc[frame_index]["pitch"],
          "roll_ground_truth": ground_truth.iloc[frame_index]["roll"],
        }
      )

      if not success:
        raise RuntimeError(f"Unable to save the frame {frame_index} in the video {video_path}")

      frame_index += 1
      running_sample_size += 1

      if total_sample_size and running_sample_size == total_sample_size:
        break

    video_capture.release()
    cv2.destroyAllWindows()

    return processed_frames

  def process_videos_for_username(
    self, username_path: Path, running_sample_size: int, total_sample_size: Optional[int] = None
  ) -> List[Dict[str, Union[str, float]]]:

    processed_frames: List[Dict[str, Union[str, float]]] = []
    for video_path in username_path.glob("*.mp4"):
      logger.info(f"Processing file {video_path}")

      ground_truth_filepath = video_path.parent / f"{video_path.stem}_groundtruth3D.txt"
      ground_truth = pd.read_csv(
        ground_truth_filepath,
        sep="\t",
        header=None,
        names=self.GROUND_TRUTH_FILE_HEADERS,
        index_col=False,
      )
      processed_frames.extend(
        self.process_single_video(
          video_path=video_path,
          ground_truth=ground_truth,
          running_sample_size=running_sample_size,
          total_sample_size=total_sample_size,
        )
      )

      if running_sample_size == total_sample_size:
        break

    return processed_frames

  def process_dataset(self, sample_size: Optional[int] = None) -> List[Dict[str, Union[str, float]]]:

    if not os.path.exists(self._images_path):
      self._images_path.mkdir(exist_ok=True)

    processed_frames: List[Dict[str, Union[str, float]]] = []
    running_sample_size = 0
    for username in os.listdir(self._dataset_path):
      processed_frames.extend(
        self.process_videos_for_username(
          username_path=self._images_path / username,
          running_sample_size=running_sample_size,
          total_sample_size=sample_size,
        )
      )

      if running_sample_size == sample_size:
        break

    return processed_frames
