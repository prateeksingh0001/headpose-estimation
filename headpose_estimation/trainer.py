from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from headpose_estimation.models.angle_prediction_heads import EulerAnglesPredictionHead
from headpose_estimation.models.pretrained_image_model import (
  PretrainedBackBoneImageModel,
)
from headpose_estimation.schema import Dataset, ExperimentConfig
from headpose_estimation.utils.constants import (
  DEFAULT_PREDICTION_HEAD_FROZEN_GRAPH_NAME,
  DEFAULT_SAVE_NP_ARRAY_NAME,
  EULER_ANGLE_NORMALIZATION_FACTOR,
  PITCH_ANGLE_KEY,
  ROLL_ANGLE_KEY,
  YAW_ANGLE_KEY,
)
from headpose_estimation.utils.tensorflow_model_handler import (
  TensorflowV1ModelConfig,
  TensorflowV1ModelHandler,
)
from headpose_estimation.utils.utils import cls_from_str, optimizer_factory, set_global_seed


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

    self._setup_experiment_directories(experiment_config=experiment_config)

    global_step = tf.Variable(initial_value=0, name="global_step", trainable=False)

    optimizer, learning_rate = optimizer_factory(self.experiment_config.optimizer_config, global_step)

    pretrained_model_config: TensorflowV1ModelConfig = TensorflowV1ModelHandler(
      model_storage_path=self.experiment_config.image_representation_model.model_dir,
    ).get_model_info(architecture=self.experiment_config.image_representation_model.architecture)

    self._pretrained_model = PretrainedBackBoneImageModel(tf_model_config=pretrained_model_config)

    self._angle_prediction_head = cls_from_str(self.experiment_config.prediction_model_config.prediction_head_class)(
      input_representation_size=pretrained_model_config.output_tensor_size,
      layer_sizes=self.experiment_config.prediction_model_config.layer_sizes,
      optimizer=optimizer,
      learning_rate_tensor=learning_rate,
      global_step=global_step,
    )

  def _setup_experiment_directories(self, experiment_config: ExperimentConfig) -> None:
    experiment_root = Path(experiment_config.experiment_root)
    if not experiment_root.exists():
      experiment_root.mkdir(exist_ok=True)

    if not experiment_config.experiment_directory.exists():
      experiment_config.experiment_directory.mkdir(parents=True, exist_ok=True)

    if not experiment_config.tensorboard_log_dir.exists():
      experiment_config.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    if not Path(experiment_config.training_config.model_save_dir).exists():
      Path(experiment_config.training_config.model_save_dir).mkdir(parents=True, exist_ok=True)

  def _load_dataset(self) -> Dataset:
    return Dataset.load_from_file(file_path=self.experiment_config.training_config.training_data_path)

  def _create_intermediate_image_representations(
    self,
    session: tf.compat.v1.Session,
    dataset: Dataset,
  ) -> Dataset:
    batch_size = self.experiment_config.training_config.batch_size
    image_representation_path = Path(self.experiment_config.training_config.intermediate_representations_save_dir)

    if not image_representation_path.exists():
      image_representation_path.mkdir(exist_ok=True)

    for index in range(0, len(dataset.datapoints), batch_size):
      # current_batch_size is used to avoid indexing errors in the cases when the number of
      # datapoints are less than than batch_size
      current_batch_size = min(batch_size, len(dataset.datapoints[index : index + batch_size]))

      raw_image_data_batch = []
      for j in range(current_batch_size):
        with tf.gfile.FastGFile(dataset[index + j].image_path, "rb") as f:
          raw_image_data_batch.append(f.read())

      image_representations: List[np.ndarray] = self._pretrained_model.predict(
        session=session, prediction_input=raw_image_data_batch
      )

      for j in range(current_batch_size):
        file_name = f"{dataset[index + j].id.split('.')[0]}.npz"
        np.savez_compressed(image_representation_path / file_name, image_representations[j])
        dataset[index + j].intermediate_representation_path = str(Path(image_representation_path) / file_name)

    return dataset

  @staticmethod
  def get_img_representation_and_gt_batch(
    dataset: Dataset, start_index: int, batch_size: int
  ) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:

    image_representations = []
    euler_angles = {YAW_ANGLE_KEY: [], PITCH_ANGLE_KEY: [], ROLL_ANGLE_KEY: []}

    for datapoint in dataset.datapoints[start_index : start_index + batch_size]:
      with np.load(datapoint.intermediate_representation_path) as data:
        image_representations.append(data[DEFAULT_SAVE_NP_ARRAY_NAME])

      euler_angles[YAW_ANGLE_KEY].append(datapoint.yaw_ground_truth)
      euler_angles[PITCH_ANGLE_KEY].append(datapoint.pitch_ground_truth)
      euler_angles[ROLL_ANGLE_KEY].append(datapoint.roll_ground_truth)

    for key, val in euler_angles.items():
      euler_angles[key] = np.array(val)

    return image_representations, euler_angles

  def calculate_validation_loss(
    self,
    session: tf.compat.v1.Session,
    dataset: Dataset,
    tensorboard_summary_tag_prefix: str = "",
  ) -> Tuple[float, float, float, float, tf.Summary]:

    batch_size = self.experiment_config.training_config.batch_size

    squared_yaw_losses = []
    squared_pitch_losses = []
    squared_roll_losses = []
    for step in range(0, len(dataset), batch_size):
      image_representations, gt_angles = self.get_img_representation_and_gt_batch(
        dataset=dataset, start_index=step, batch_size=batch_size
      )

      angle_predictions = self._angle_prediction_head.predict(session=session, prediction_input=image_representations)

      for predicted_angles, yaw_gt, roll_gt, pitch_gt in zip(
        angle_predictions,
        gt_angles[YAW_ANGLE_KEY],
        gt_angles[ROLL_ANGLE_KEY],
        gt_angles[PITCH_ANGLE_KEY],
      ):
        squared_yaw_losses.append(((predicted_angles[YAW_ANGLE_KEY] - yaw_gt) / EULER_ANGLE_NORMALIZATION_FACTOR) ** 2)
        squared_roll_losses.append(
          ((predicted_angles[ROLL_ANGLE_KEY] - roll_gt) / EULER_ANGLE_NORMALIZATION_FACTOR) ** 2
        )
        squared_pitch_losses.append(
          ((predicted_angles[PITCH_ANGLE_KEY] - pitch_gt) / EULER_ANGLE_NORMALIZATION_FACTOR) ** 2
        )

    yaw_mse = np.mean(squared_yaw_losses)
    roll_mse = np.mean(squared_roll_losses)
    pitch_mse = np.mean(squared_pitch_losses)

    total_mse = np.mean([yaw_mse, roll_mse, pitch_mse])

    tensorboard_val_summary = tf.Summary(
      value=[
        tf.Summary.Value(
          tag=f"{tensorboard_summary_tag_prefix}/total_loss",
          simple_value=total_mse,
        ),
        tf.Summary.Value(
          tag=f"{tensorboard_summary_tag_prefix}/yaw_loss",
          simple_value=yaw_mse,
        ),
        tf.Summary.Value(
          tag=f"{tensorboard_summary_tag_prefix}/roll_loss",
          simple_value=roll_mse,
        ),
        tf.Summary.Value(
          tag=f"{tensorboard_summary_tag_prefix}/pitch_loss",
          simple_value=pitch_mse,
        ),
      ]
    )

    return total_mse, yaw_mse, roll_mse, pitch_mse, tensorboard_val_summary

  def train(self) -> None:
    set_global_seed(self.experiment_config.seed)

    batch_size = self.experiment_config.training_config.batch_size
    num_epochs = self.experiment_config.training_config.num_epochs
    prediction_head_save_path = str(
      Path(self.experiment_config.training_config.model_save_dir) / DEFAULT_PREDICTION_HEAD_FROZEN_GRAPH_NAME
    )

    dataset = self._load_dataset()

    with tf.compat.v1.Session() as session:
      if self.experiment_config.training_config.run_bottleneck_generations:
        dataset: Dataset = self._create_intermediate_image_representations(session=session, dataset=dataset)
        dataset.save(save_path=self.experiment_config.training_config.training_data_path)

      train_dataset, val_dataset, test_dataset = dataset.shuffle_and_train_test_split(
        train_percentage=self.experiment_config.training_config.train_percentage,
        validation_percentage=self.experiment_config.training_config.validation_percentage,
        test_percentage=self.experiment_config.training_config.test_percentage,
      )

      session.run(tf.global_variables_initializer())

      tensorboard_summary_writer = tf.summary.FileWriter(
        logdir=self.experiment_config.tensorboard_log_dir, graph=session.graph
      )

      training_total_loss = []
      training_yaw_loss = []
      training_pitch_loss = []
      training_roll_loss = []
      validation_losses = []

      for epoch in range(num_epochs):
        for step in range(0, len(train_dataset), batch_size):
          global_step = (epoch * len(train_dataset)) + step

          if global_step and (global_step % self.experiment_config.training_config.eval_step_interval == 0):
            total_val_loss, _, _, _, val_summary = self.calculate_validation_loss(
              session=session,
              dataset=val_dataset,
              tensorboard_summary_tag_prefix="validation",
            )
            validation_losses.append(total_val_loss)
            tensorboard_summary_writer.add_summary(val_summary, global_step=global_step)

          image_representations, ground_truth = self.get_img_representation_and_gt_batch(
            dataset=train_dataset,
            start_index=step,
            batch_size=batch_size,
          )

          total_loss, yaw_loss, roll_loss, pitch_loss, summary = self._angle_prediction_head.train(
            session=session,
            image_representations=image_representations,
            ground_truths=ground_truth,
          )

          tensorboard_summary_writer.add_summary(summary, global_step=global_step)

          training_total_loss.append(total_loss)
          training_yaw_loss.append(yaw_loss)
          training_pitch_loss.append(pitch_loss)
          training_roll_loss.append(roll_loss)
          print(
            f"\nGS: {global_step}, TL: {training_total_loss[-1]}, YL: {training_yaw_loss[-1]}, PL: {training_pitch_loss[-1]}, RL: {training_roll_loss[-1]}"
          )

      test_loss, _, _, _, test_summary = self.calculate_validation_loss(
        session=session,
        dataset=test_dataset,
        tensorboard_summary_tag_prefix="test",
      )
      tensorboard_summary_writer.add_summary(test_summary, global_step=0)
      print(f"\n Test Loss: {test_loss}")

      self._angle_prediction_head.save_as_frozen_graph(session=session, output_graphdef_path=prediction_head_save_path)
