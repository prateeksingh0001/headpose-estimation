from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from headpose_estimation.models import (
    EulerAnglesPredictionHead,
    PretrainedBackBoneImageModel,
)
from headpose_estimation.schema import Dataset, ExperimentConfig
from headpose_estimation.utils.tensorflow_model_handler import (
    TensorflowV1ModelConfig,
    TensorflowV1ModelHandler,
)
from headpose_estimation.utils.utils import set_global_seed


class PredictionHeadTrainer:
    """
    A trainer class that orchestrates and an end-to-end training experiment.

    Steps:
        1. Setup up the backbone image model and the angle prediction heads.
        2. Generate intermediate representations using the backbone model and cached them on disk.
        3. Load the intermediate representations and use them to train the prediction head.
        4. Save a combined model on the disk.
            - Input: Image bytes as a string
            - Ouput: Prediction angles
    """

    def __init__(self, experiment_config: ExperimentConfig) -> None:
        self.experiment_config = experiment_config

        pretrained_model_config: TensorflowV1ModelConfig = TensorflowV1ModelHandler(
            tf_model_config_path="",
            model_storage_path=self.experiment_config.image_representation_model.model_dir,
        ).get_model_info(
            architecture=self.experiment_config.image_representation_mdoel.architecture
        )

        self._pretrained_model = PretrainedBackBoneImageModel(
            tf_model_config=pretrained_model_config
        )

        self._angle_prediction_head = EulerAnglesPredictionHead(
            input_representation_size=pretrained_model_config.output_tensor_size,
            layer_sizes=self.experiment_config.prediction_model_config.layer_sizes,
            optimizer=None,
        )

    def _load_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        dataset = Dataset.load_from_file(
            file_path=self.experiment_config.training_config.training_data_path
        )
        train_dataset, validation_dataset, test_dataset = (
            dataset.shuffle_and_train_test_split(
                train_percentage=self.experiment_config.training_config.train_percentage,
                validation_percentage=self.experiment_config.training_config.validation_percentage,
                test_percentage=self.experiment_config.training_config.test_percentage,
            )
        )

        return train_dataset, validation_dataset, test_dataset

    def _create_intermediate_image_representations(
        self,
        session: tf.Session,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        test_dataset: Dataset,
    ) -> None:
        batch_size = self.experiment_config.training_config.batch_size
        image_representation_path = Path(
            self.experiment_config.training_config.intermediate_representations_save_dir
        )

        for dataset in [train_dataset, validation_dataset, test_dataset]:
            for index in range(0, len(dataset.datapoints), batch_size):
                raw_image_data_batch = []
                for j in range(batch_size):
                    with tf.gfile.FastGFile(dataset.datapoints[index + j], "rb") as f:
                        raw_image_data_batch.append(f.read())

                image_representations: List[np.ndarray] = (
                    self._pretrained_model.predict(
                        session=session, prediction_input=raw_image_data_batch
                    )
                )

                for j in range(batch_size):
                    file_name = f"{dataset.datapoints[index + j].id.split('.')[0]}.npz"
                    np.savez_compressed(
                        image_representation_path / file_name, image_representations[j]
                    )
                    dataset.datapoints[
                        index + j
                    ].intermediate_representatino_path = str(
                        Path(image_representation_path) / file_name
                    )

    @staticmethod
    def get_img_representation_and_gt_batch(
        dataset: Dataset, start_index: int, batch_size: int
    ) -> Tuple[List[np.ndarray], Dict[str, List[float]]]:

        image_representations = []
        euler_angles = {"yaw": [], "pitch": [], "roll": []}

        for datapoint in dataset.datapoints[start_index : start_index + batch_size]:
            image_representations.append(
                np.load(datapoint.intermediate_representation_path)
            )
            euler_angles["yaw"].append(datapoint.yaw_ground_truth)
            euler_angles["pitch"].append(datapoint.pitch_ground_truth)
            euler_angles["roll"].append(datapoint.roll_ground_truth)

        return image_representations, euler_angles

    def calculate_validation_loss(
        self, session: tf.Session, val_dataset: Dataset
    ) -> float:
        batch_size = self.experiment_config.training_config.batch_size

        per_batch_val_loss = []
        for step in range(0, len(val_dataset), batch_size):
            image_representations, gt_angles = self.get_img_representation_and_gt_batch(
                dataset=val_dataset, start_index=step, batch_size=batch_size
            )

            angle_predictions = self._angle_prediction_head.predict(
                session=session, prediction_input=image_representations
            )

            per_example_loss = []
            for predicted_angles, ground_truth in zip(angle_predictions, gt_angles):
                yaw_loss = predicted_angles["yaw"] - ground_truth["yaw"]
                roll_loss = predicted_angles["roll"] - ground_truth["roll"]
                pitch_loss = predicted_angles["pitch"] - ground_truth["pitch"]
                per_example_loss.append([yaw_loss + roll_loss + pitch_loss])

            per_batch_val_loss.append(sum(per_example_loss))

        return sum(per_batch_val_loss)

    def train(self) -> None:
        set_global_seed(self.experiment_config.random_seed)

        batch_size = self.experiment_config.training_config.batch_size
        num_epochs = self.experiment_config.training_config.num_epochs
        prediction_head_save_path = str(
            Path(self.experiment_config.training_config.model_save_dir)
            / "prediction_head.pb"
        )

        train_dataset, val_dataset, test_dataset = self._load_dataset()

        with tf.Session() as session:
            session.run(tf.global_variable_initializer())

            self._create_intermediate_image_representations(
                session=session,
                train_dataset=train_dataset,
                validation_dataset=val_dataset,
                test_dataset=test_dataset,
            )

            training_total_loss = []
            training_yaw_loss = []
            training_pitch_loss = []
            training_roll_loss = []
            validation_losses = []

            for epoch in range(num_epochs):
                for step in range(0, len(train_dataset), batch_size):
                    if (
                        step % self.experiment_config.training_config.eval_step_interval
                        == 0
                    ):
                        val_loss = self.calculate_validation_loss(
                            session=session, dataset=val_dataset
                        )
                        validation_losses.append(val_loss)

                    image_representations, ground_truth = (
                        self.get_img_representation_and_gt_batch(
                            dataset=train_dataset,
                            start_index=step,
                            batch_size=batch_size,
                        )
                    )
                    total_loss, yaw_loss, pitch_losss, roll_loss = (
                        self._angle_prediction_head.train(
                            sesssion=session,
                            image_representation=image_representations,
                            ground_truths=ground_truth,
                        )
                    )

                    training_total_loss.append(total_loss)
                    training_yaw_loss.append(yaw_loss)
                    training_pitch_loss.append(pitch_losss)
                    training_roll_loss.append(roll_loss)

            test_loss = self.calculate_validation_loss(
                session=session, dataset=test_dataset
            )
            self._angle_prediction_head.save_as_frozen_graph(
                session=session, output_graphdef_path=prediction_head_save_path
            )
