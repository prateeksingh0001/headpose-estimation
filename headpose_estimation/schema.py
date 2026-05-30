from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Tuple, Union

import yaml
from yaml import Loader


@dataclass
class Datapoint:
    id: str
    image_path: str
    yaw_ground_truth: float
    pitch_ground_truth: float
    roll_ground_truth: float
    intermediate_representation_path: str = None


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
        self,
        train_percentage: int,
        validation_percentage: int,
        test_percentage: int,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        random.shuffle(self.datapoints)
        training_data_size = ceil(len(self.datapoints) * train_percentage / 100)
        validation_data_size = ceil(len(self.datapoints) * validation_percentage / 100)
        test_data_size = ceil(len(self.datapoints) * test_percentage / 100)

        train_dataset = Dataset(datapoints=self.datapoints[:training_data_size])
        validation_dataset = Dataset(
            datapoints=self.datapoints[: (training_data_size + validation_data_size)]
        )
        test_dataset = Dataset(
            datapoints=self.datapoints[
                : (training_data_size + validation_data_size + test_data_size)
            ]
        )

        return train_dataset, validation_dataset, test_dataset


@dataclass
class TrainingConfig:
    training_data_path: str
    train_percentage: float = 80.0
    validation_percentage: float = 10.0
    test_percentage: float = 10.0
    intermediate_representations_save_dir: str
    num_epochs: int
    batch_size: int
    eval_step_interval: int
    model_save_dir: str
    checkpoint_save_frequency: int = 100


@dataclass
class PretrainedImageRepresentationModelConfig:
    architecture: str
    model_dir: str


@dataclass
class PredictionHeadModelConfig:
    layer_sizes: List[int]


@dataclass
class OptimizerConfig:
    optimizer_class: str = "tf.train.AdamOptimizer"
    learning_rate: float = "0.01"
    optimizer_params: Dict[str, float] = {}
    learning_rate_decay_function: str = None
    decay_steps: int = None
    decay_rate: float = None


@dataclass
class ExperimentConfig:
    """ """

    image_representation_model: PretrainedImageRepresentationModelConfig
    prediction_model_config: PredictionHeadModelConfig
    optimizer_config: OptimizerConfig
    training_config: TrainingConfig
    random_seed: int

    @classmethod
    def from_dict(
        cls, config_dict: Dict[str, Dict[str, Union[str, int, float, List[int]]]]
    ) -> ExperimentConfig:
        image_representation_model_details = PretrainedImageRepresentationModelConfig(
            **config_dict["image_representation_model"]
        )
        prediction_model_config = PredictionHeadModelConfig(
            **config_dict["prediction_head"]
        )
        optimizer_config = OptimizerConfig(**config_dict["optimizer_config"])
        training_config = TrainingConfig(**config_dict["training_config"])
        return ExperimentConfig(
            image_representation_model=image_representation_model_details,
            prediction_model_config=prediction_model_config,
            optimizer_config=optimizer_config,
            training_config=training_config,
            random_seed=config_dict["random_seed"],
        )

    @classmethod
    def from_yaml(cls, config_path: str) -> ExperimentConfig:
        with open(config_path) as f:
            config_dict: Dict[str, Union[str, int]] = yaml.load(f, Loader=Loader)

        image_representation_model_details = PretrainedImageRepresentationModelConfig(
            **config_dict["image_representation_model"]
        )
        prediction_model_config = PredictionHeadModelConfig(
            **config_dict["prediction_head"]
        )
        optimizer_config = OptimizerConfig(**config_dict["optimizer_config"])
        training_config = TrainingConfig(**config_dict["training_config"])
        return ExperimentConfig(
            image_representation_model=image_representation_model_details,
            prediction_model_config=prediction_model_config,
            optimizer_config=optimizer_config,
            training_config=training_config,
            random_seed=config_dict["random_seed"],
        )

    @classmethod
    def from_args(cls) -> ExperimentConfig:
        args = cls._parse_args()
        config_path = args.pop("config_path")

        if config_path:
            return cls.from_yaml(config_path=config_path)
        else:
            image_representation_model_details = (
                PretrainedImageRepresentationModelConfig(
                    architecture=args["architecture"], model_dir=args["model_dir"]
                )
            )

            prediction_model_config = PredictionHeadModelConfig(
                layer_sizes=args["layer_sizes"]
            )

            if "optimizer_class" in args and "learning_rate" in args:
                optimizer_config = OptimizerConfig(
                    optimizer_class=args["optimizer_class"],
                    learning_rate=args["learning_rate"],
                )
                if "optimizer_params" in args:
                    optimizer_config.optimizer_params = args["optimizer_params"]

                if "learning_rate_decay_function" in args:
                    optimizer_config.learning_rate_decay_function = args[
                        "learning_rate_decay_function"
                    ]
                    optimizer_config.decay_steps = args["decay_steps"]
                    optimizer_config.decay_rate = args["decay_rate"]

            training_config = TrainingConfig(
                training_data_path=args["training_data_path"],
                train_percentage=args["train_percentage"],
                test_percentage=args["test_percentage"],
                intermediate_representations_save_dir=args[
                    "intermediate_representation_save_dir"
                ],
                num_epochs=args["num_epochs"],
                batch_size=args["batch_size"],
                eval_step_interval=args["eval_step_interval"],
                model_save_dir=args["model_save_dir"],
                checkpoint_save_frequency=args["checkpoint_save_frequency"],
            )
            return cls(
                image_representation_model_details=image_representation_model_details,
                prediction_model_config=prediction_model_config,
                optimizer_config=optimizer_config,
                training_config=training_config,
                random_seed=args["random_seed"],
            )

    @staticmethod
    def _parse_args() -> Dict[str, Union[str, int, float]]:
        parser = argparse.ArgumentParser()

        # Config files
        parser.add_argument("-c", "--config-path", type=str)

        # Pretrained Image backbone params
        parser.add_argument(
            "--architecture",
            type=str,
            dest="architecture",
            help="architecture to be used in the frontend",
        )
        parser.add_argument(
            "--bottleneck-model-dir",
            type=str,
            dest="model_dir",
            help="The directory where the model is to be saved",
        )

        # Prediction head params
        parser.add_argument(
            "--prediction-head-layer-sizes",
            nargs="+",
            type=lambda s: [int(item) for item in s.split(",")],
            default=[],
            dest="layer_sizes",
            help="size of the intermediate layers in the angle prediction head MLP as comma seperated values.",
        )

        # Optimizer config params
        parser.add_argument(
            "--optimizer",
            type=str,
            dest="optimizer_class",
            help="Reference path to the optimizer class",
        )
        parser.add_argument(
            "--learning-rate",
            type=float,
            default=0.01,
            dest="learning_rate",
            help="Learning rate for the algo",
        )
        parser.add_argument(
            "--learning-rate-decay-function",
            type=str,
            dest="learning_rate_decay_function",
            help="The function to use for decaying learning rate",
        )
        parser.add_argument(
            "--decay-steps",
            type=int,
            dest="decay_steps",
            help="Number of step after trigger the learning rate decay",
        )
        parser.add_argument(
            "--decay-rate",
            type=float,
            dest="decay_rate",
            help="Rate at which to decay the learning rate at",
        )

        # Training config
        parser.add_argument(
            "--training-data-path",
            type=str,
            dest="training_data_path",
            help="The directory to store images",
        )
        parser.add_argument(
            "--train-percentage",
            type=float,
            dest="train_percentage",
            help="The path to the file with all the labels",
        )
        parser.add_argument(
            "--validation-percentage",
            type=float,
            dest="validation_precentage",
            help="The path to the file with all the labels",
        )
        parser.add_argument(
            "--test-percentage",
            type=float,
            dest="test_percentage",
            help="The path to the file with all the labels",
        )
        parser.add_argument(
            "--intermediate-representation-save-dir",
            type=str,
            dest="intermediate_represenation_save_dir",
            help="Directory to store the intermediate representations created by the pretrained backbone image model",
        )
        parser.add_argument(
            "--num-epochs",
            type=int,
            dest="num_epochs",
            help="number of epoch to train the prediction head on",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=5,
            dest="batch_size",
            help="Batch size for training, validation and testing",
        )
        parser.add_argument(
            "--eval-step-interval",
            type=int,
            default=10,
            dest="eval_step_interval",
            help="How often to evaluate the training results",
        )
        parser.add_argument(
            "--model-save-dir",
            type=str,
            dest="model_save_dir",
            help="Where to save the exported graphs",
        )
        parser.add_argument(
            "--checkpoint-save-frequency",
            default=100,
            type=int,
            dest="checkpoint_save_frequency",
            help="When to store the intermediate graphs",
        )

        # random seed
        parser.add_argument(
            "--random-seed",
            type=int,
            dest="random_seed",
            default=100,
            help="Seed to be used for any random number invocations for example weight initialization, dataset"
            " shuffling etc.",
        )

        args = parser.parse_args()
        return vars(args)
