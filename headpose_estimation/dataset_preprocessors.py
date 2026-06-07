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
import math
import os
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

ProcessedFrameType = List[Dict[str, Union[str, float]]]


class BaseDatasetPreprocessor:
  def __init__(self, dataset_name: str, dataset_path: str, output_path: str):
    self._dataset_name = dataset_name
    self._dataset_path = Path(dataset_path)
    self._images_path = Path(output_path)

  @abstractmethod
  def process_dataset(self, sample_size: Optional[int] = None) -> ProcessedFrameType:
    pass


class UPNAPreprocessor(BaseDatasetPreprocessor):
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

  def process_single_video(
    self,
    video_path: Path,
    ground_truth: pd.DataFrame,
    running_sample_size: int,
    total_sample_size: Optional[int] = None,
  ) -> ProcessedFrameType:

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

    return processed_frames, running_sample_size

  def process_videos_for_username(
    self, username_path: Path, running_sample_size: int, total_sample_size: Optional[int] = None
  ) -> ProcessedFrameType:

    processed_frames: ProcessedFrameType = []
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
      single_video_processed_frames, running_sample_size = self.process_single_video(
        video_path=video_path,
        ground_truth=ground_truth,
        running_sample_size=running_sample_size,
        total_sample_size=total_sample_size,
      )

      processed_frames.extend(single_video_processed_frames)

      if total_sample_size and running_sample_size == total_sample_size:
        break

    return processed_frames, running_sample_size

  def process_dataset(self, sample_size: Optional[int] = None) -> ProcessedFrameType:

    processed_frames: ProcessedFrameType = []
    running_sample_size = 0
    for username in os.listdir(self._dataset_path):
      user_processed_frames, running_sample_size = self.process_videos_for_username(
        username_path=self._dataset_path / username,
        running_sample_size=running_sample_size,
        total_sample_size=sample_size,
      )

      processed_frames.extend(user_processed_frames)

      if sample_size and running_sample_size == sample_size:
        break

    return processed_frames


class ThreeHudredWLPPreprocessor(BaseDatasetPreprocessor):
  """
  Preprocessor for the 300WLP dataset. The dataset consists of multiple subdatasets list in the class variable
  `SUB_DATASET_NAMES`.
  Each subdataset consists of multiple image and a matlab data files(.mat) associated with each image, each matlab
  file consists of the following keys:
  - pt2d
  - roi
  - Illum_Para
  - Color_Para
  - Tex_Para
  - Shape_Para
  - Exp_Para
  - Pose_Para

  The Pose_Para key contains 6 values the first three of which are Pitch, Yaw and Roll
  """

  SUB_DATASET_NAMES = ["AFW", "AFW_Flip", "HELEN", "HELEN_Flip", "IBUG", "IBUG_Flip", "LFPW", "LFPW_Flip"]

  def process_sub_dataset(
    self, sub_dataset_path: Path, running_sample_size: int, sample_size: Optional[int] = None
  ) -> ProcessedFrameType:
    processed_frames: ProcessedFrameType = []
    for image_path in sub_dataset_path.glob("*.jpg"):
      image_file_name = image_path.stem
      image_data = loadmat(image_path.with_suffix(".mat"))

      # Convert radians to degrees
      pitch, yaw, roll = map(lambda x: math.degrees(x), image_data["Pose_Para"][0, :3])

      processed_frames.append(
        {
          "id": str(image_file_name),
          "image_path": str(image_path),
          "yaw_ground_truth": yaw,
          "pitch_ground_truth": pitch,
          "roll_ground_truth": roll,
        }
      )

      running_sample_size += 1

      if sample_size and running_sample_size == sample_size:
        break

    return processed_frames, running_sample_size

  def process_dataset(self, sample_size: Optional[int] = None) -> ProcessedFrameType:
    running_sample_size = 0
    processed_dataset: ProcessedFrameType = []
    for sub_dataset_name in self.SUB_DATASET_NAMES:
      sub_dataset_path = self._dataset_path / sub_dataset_name
      processed_sub_dataset, running_sample_size = self.process_sub_dataset(
        sub_dataset_path=sub_dataset_path,
        running_sample_size=running_sample_size,
        sample_size=sample_size,
      )

      processed_dataset.extend(processed_sub_dataset)

      if sample_size and running_sample_size == sample_size:
        break

    return processed_dataset
