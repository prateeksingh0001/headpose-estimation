import csv
import argparse
import logging
from pathlib import Path

from scipy.io import loadmat

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder")
    parser.add_argument("--output")

    return parser.parse_args()


def main() -> None:

    args = parse_args()
    img_metadata_paths = list(Path(args.input_folder).rglob("*.mat"))

    ground_truths = []
    for img_metadata_path in img_metadata_paths:
        img_data = loadmat(img_metadata_path)
        try:
            pitch, yaw, roll = img_data["Pose_Para"][0, 0:3]
            yaw = -yaw

            ground_truths.append((pitch, yaw, roll))
        except KeyError as e:
            logger.warning(f"Did not find key Pose_Para in {img_metadata_path}") 

    with open(args.output, "w") as ground_truth_file:
        tsvwriter = csv.writer(ground_truth_file, delimiter="\t")
        tsvwriter.writerows(ground_truths) 
    

if __name__ == "__main__":
    main()
