from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import yaml
from yaml import Loader


@dataclass
class Datapoint:
    id: str
    image: str
    yaw_ground_truth: float
    pitch_ground_truth: float
    roll_ground_truth: float


@classmethod
class Dataset:
    datapoints: List[Datapoint]

    @classmethod
    def load_from_file(cls, file_path: str) -> Dataset:
        with open(file_path) as f:
            data = json.load(f)

        dataset = [Datapoint(*datapoint) for datapoint in data]

        return cls(datapoints=dataset)

    def shuffle_and_train_test_split(
        self, split_config: DataDistributionConfig
    ) -> Tuple[Dataset, Dataset, Dataset]:
        pass


@dataclass
class DataDistributionConfig:
    train_percentage: float = 80.0
    validation_percentage: float = 10.0
    test_percentage: float = 10.0


@dataclass
class TrainingExperimentConfig:
    architecture: str
    data_distribution_config: DataDistributionConfig
    model_dir: str
    image_dir: str
    labels_file: str
    summaries_dir: str
    bottleneck_dir: str
    saved_model_dir: str
    num_epochs: int
    output_graph: str
    intermediate_output_graph_dir: str = "/tmp/intermediate_graphs/"
    train_batch_size: int = 5
    test_batch_size: int = 5
    validation_batch_size: int = 5
    learning_rate: float = 0.01
    intermediate_store_frequency: int = 100
    eval_step_interval: int = 10
    final_tensor_name: str = "final_tensor_name"
    random_seed: int = 0

    @classmethod
    def from_yaml(cls, config_path: str) -> TrainingExperimentConfig:
        with open(config_path) as f:
            config_dict: Dict[str, Union[str, int]] = yaml.load(f, Loader=Loader)

        return cls(**config_dict)

    @classmethod
    def from_args(cls) -> TrainingExperimentConfig:
        args = cls._parse_args()
        config_path = args.pop("config_path")

        if config_path:
            return cls.from_yaml(config_path=config_path)
        else:
            return TrainingExperimentConfig(**args)

    @staticmethod
    def _parse_args() -> Dict[str, Union[str, int]]:
        parser = argparse.ArgumentParser()

        parser.add_argument("-c", "--config-path", type=str)
        parser.add_argument(
            "--architecture",
            type=str,
            dest="architecture",
            help="architecture to be used in the frontend",
        )
        parser.add_argument(
            "--model_dir",
            type=str,
            dest="model_dir",
            help="The directory where the model is to be saved",
        )
        parser.add_argument(
            "--image_dir",
            type=str,
            dest="image_dir",
            help="The directory to store images",
        )
        parser.add_argument(
            "--labels_file",
            type=str,
            dest="labels_file",
            help="The path to the file with all the labels",
        )
        parser.add_argument(
            "--summaries_dir",
            type=str,
            dest="summaries_dir",
            help="Directory for saving the summaries",
        )
        parser.add_argument(
            "--num-epoch",
            type=int,
            dest="num_epochs",
            help="Num of epochs to train the model on",
        )
        parser.add_argument(
            "--validation_percentage",
            type=int,
            dest="validation_percentage",
            help="percentage of the dataset that is to be used for validation set",
        )
        parser.add_argument(
            "--testing_percentage",
            type=int,
            dest="testing_percentage",
            help="percentage of dataset that is to be used as test set",
        )
        parser.add_argument(
            "--bottleneck_dir",
            type=str,
            dest="bottleneck_dir",
            help="Directory where the bottleneck tensors are stored",
        )
        parser.add_argument(
            "--saved_model_dir",
            type=str,
            dest="saved_model_dir",
            help="Where to save the exported graph",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=0.01,
            dest="learning_rate",
            help="Learning rate for the algo",
        )
        parser.add_argument(
            "--train_batch_size",
            type=int,
            default=5,
            dest="train_batch_size",
            help="Batch size for training",
        )
        parser.add_argument(
            "--validation_batch_size",
            type=int,
            default=5,
            dest="validation_batch_size",
            help="Batch size for validation",
        )
        parser.add_argument(
            "--test_batch_size",
            type=int,
            default=5,
            dest="test_batch_size",
            help="Batch size for testing",
        )
        parser.add_argument(
            "--final_tensor_name",
            type=str,
            default="final_tensor_name",
            help="""The name of the output classification layer in the retrained graph.""",
        )
        parser.add_argument(
            "--output_graph",
            type=str,
            default="/tmp/retrained_graph.pb",
            help="""The name of the output classification layer in the retrained graph.""",
        )
        parser.add_argument(
            "--intermediate_store_frequency",
            default=100,
            type=int,
            dest="intermediate_store_frequency",
            help="When to store the intermediate graphs",
        )
        parser.add_argument(
            "--intermediate_output_graph_dir",
            type=str,
            default="/tmp/intermediate_graphs/",
            dest="intermediate_output_graph_dir",
            help="Where to stor the intermediate graphs",
        )
        parser.add_argument(
            "--eval_step_interval",
            type=int,
            default=10,
            dest="eval_step_interval",
            help="""How often to evaluate the training results""",
        )

        args = parser.parse_args()
        return vars(args)
