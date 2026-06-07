from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from headpose_estimation.schema import (
  Datapoint,
  Dataset,
  ExperimentConfig,
  OptimizerConfig,
  PredictionHeadModelConfig,
  PretrainedImageRepresentationModelConfig,
  TrainingConfig,
)
from headpose_estimation.trainer import PredictionHeadTrainer
from headpose_estimation.utils.constants import (
  PITCH_ANGLE_KEY,
  ROLL_ANGLE_KEY,
  YAW_ANGLE_KEY,
)

FIXTURE_IMAGES = [
  "tests/headpose_estimation/fixtures/images/AFW_134212_1_0.jpg",
  "tests/headpose_estimation/fixtures/images/IBUG_image_011_1_2.jpg",
  "tests/headpose_estimation/fixtures/images/LFPW_image_test_0001_3.jpg",
]


class _RepresentationEchoHead:
  """Stub prediction head that predicts each representation's scalar value as the angle."""

  def predict(self, session, prediction_input):
    return [
      {
        YAW_ANGLE_KEY: float(representation.flat[0]),
        PITCH_ANGLE_KEY: float(representation.flat[0]),
        ROLL_ANGLE_KEY: float(representation.flat[0]),
      }
      for representation in prediction_input
    ]


class TestPredictionHeadTrainer:
  @pytest.fixture
  def experiment_config(self, tmp_path) -> ExperimentConfig:
    return ExperimentConfig(
      experiment_name="test_experiment",
      experiment_root=str(tmp_path / "experiments"),
      image_representation_model=PretrainedImageRepresentationModelConfig(
        architecture="mock_backbone", model_dir=str(tmp_path / "models")
      ),
      prediction_model_config=PredictionHeadModelConfig(layer_sizes=[4]),
      optimizer_config=OptimizerConfig(),
      training_config=TrainingConfig(
        training_data_path=str(tmp_path / "dataset.json"),
        intermediate_representations_save_dir=str(tmp_path / "bottlenecks"),
        num_epochs=1,
        batch_size=2,
        eval_step_interval=10,
        model_save_dir=str(tmp_path / "saved_model"),
        checkpoint_save_frequency=100,
      ),
      seed=0,
    )

  @staticmethod
  def _bare_trainer(experiment_config: ExperimentConfig) -> PredictionHeadTrainer:
    # Bypass __init__ (which loads a real backbone from the registry) so we can unit
    # test individual methods in isolation.
    trainer = PredictionHeadTrainer.__new__(PredictionHeadTrainer)
    trainer.experiment_config = experiment_config
    return trainer

  @staticmethod
  def _write_representation(path: Path, value: float, size: int = 4) -> None:
    # np.savez_compressed stores a single unnamed array under the key "arr_0",
    # which is what the trainer reads back (DEFAULT_SAVE_NP_ARRAY_NAME).
    np.savez_compressed(path, np.full((1, size), value, dtype=np.float32))

  @classmethod
  def _dataset_with_cached_representations(
    cls, tmp_path: Path, count: int, ground_truth_offset: float = 0.0
  ) -> Dataset:
    """Build a Dataset whose datapoints point at on-disk .npz representations.

    Datapoint ``i`` caches a representation of all-``i`` and has ground truth angles
    of ``i + ground_truth_offset`` on every axis, so a predictor that echoes the
    representation matches the ground truth exactly when the offset is zero.
    """
    datapoints = []
    for index in range(count):
      representation_path = tmp_path / f"representation_{index}.npz"
      cls._write_representation(representation_path, float(index))
      datapoints.append(
        Datapoint(
          id=f"{index}.jpg",
          image_path="unused",
          yaw_ground_truth=float(index) + ground_truth_offset,
          pitch_ground_truth=float(index) + ground_truth_offset,
          roll_ground_truth=float(index) + ground_truth_offset,
          intermediate_representation_path=str(representation_path),
        )
      )
    return Dataset(datapoints=datapoints)

  # --- get_img_representation_and_gt_batch ---

  def test_get_batch_loads_representations_and_stacks_ground_truths(self, tmp_path):
    dataset = self._dataset_with_cached_representations(tmp_path, count=5)

    representations, ground_truths = PredictionHeadTrainer.get_img_representation_and_gt_batch(
      dataset=dataset, start_index=1, batch_size=2
    )

    assert len(representations) == 2
    assert representations[0].shape == (1, 4)
    assert np.allclose(representations[0], 1.0)
    assert np.allclose(representations[1], 2.0)

    for key in (YAW_ANGLE_KEY, PITCH_ANGLE_KEY, ROLL_ANGLE_KEY):
      assert isinstance(ground_truths[key], np.ndarray)
      assert ground_truths[key].shape == (2,)
    assert np.allclose(ground_truths[YAW_ANGLE_KEY], [1.0, 2.0])

  def test_get_batch_handles_partial_final_batch(self, tmp_path):
    dataset = self._dataset_with_cached_representations(tmp_path, count=3)

    representations, ground_truths = PredictionHeadTrainer.get_img_representation_and_gt_batch(
      dataset=dataset, start_index=2, batch_size=5
    )

    assert len(representations) == 1
    assert ground_truths[YAW_ANGLE_KEY].shape == (1,)

  # --- _setup_experiment_directories ---

  def test_setup_creates_all_experiment_directories(self, experiment_config):
    trainer = self._bare_trainer(experiment_config)

    trainer._setup_experiment_directories(experiment_config)

    assert Path(experiment_config.experiment_root).exists()
    assert experiment_config.experiment_directory.exists()
    assert experiment_config.tensorboard_log_dir.exists()
    assert Path(experiment_config.training_config.model_save_dir).exists()

  def test_setup_is_idempotent_when_directories_exist(self, experiment_config):
    trainer = self._bare_trainer(experiment_config)

    trainer._setup_experiment_directories(experiment_config)
    trainer._setup_experiment_directories(experiment_config)

    assert experiment_config.experiment_directory.exists()

  # --- _load_dataset ---

  def test_load_dataset_round_trips_saved_manifest(self, experiment_config):
    original = Dataset(
      datapoints=[
        Datapoint(id="0.jpg", image_path="a", yaw_ground_truth=1.0, pitch_ground_truth=2.0, roll_ground_truth=3.0)
      ]
    )
    original.save(experiment_config.training_config.training_data_path)
    trainer = self._bare_trainer(experiment_config)

    loaded = trainer._load_dataset()

    assert len(loaded) == 1
    assert loaded[0].yaw_ground_truth == 1.0
    assert loaded[0].roll_ground_truth == 3.0

  # --- calculate_validation_loss ---

  def test_validation_loss_is_zero_when_predictions_match_ground_truth(self, experiment_config, tmp_path):
    dataset = self._dataset_with_cached_representations(tmp_path, count=4)
    trainer = self._bare_trainer(experiment_config)
    trainer._angle_prediction_head = _RepresentationEchoHead()

    total, yaw, roll, pitch, _ = trainer.calculate_validation_loss(session=None, dataset=dataset)

    assert total == pytest.approx(0.0)
    assert yaw == pytest.approx(0.0)
    assert roll == pytest.approx(0.0)
    assert pitch == pytest.approx(0.0)

  def test_validation_loss_returns_positive_loss_and_tagged_summary(self, experiment_config, tmp_path):
    # Ground truth is offset from the echoed prediction by exactly one normalization
    # factor (90 deg), so each squared, normalized error is ((-90)/90)**2 == 1.
    dataset = self._dataset_with_cached_representations(tmp_path, count=4, ground_truth_offset=90.0)
    trainer = self._bare_trainer(experiment_config)
    trainer._angle_prediction_head = _RepresentationEchoHead()

    total, _, _, _, summary = trainer.calculate_validation_loss(
      session=None, dataset=dataset, tensorboard_summary_tag_prefix="validation"
    )

    assert total == pytest.approx(1.0)
    summary_tags = {value.tag for value in summary.value}
    assert summary_tags == {
      "validation/total_loss",
      "validation/yaw_loss",
      "validation/roll_loss",
      "validation/pitch_loss",
    }

  # --- _create_intermediate_image_representations ---

  def test_create_intermediate_representations_caches_one_npz_per_datapoint(self, experiment_config, mock_backbone):
    trainer = self._bare_trainer(experiment_config)
    trainer._pretrained_model = mock_backbone.build_model()
    dataset = Dataset(
      datapoints=[
        Datapoint(
          id=f"sample_{index}.jpg",
          image_path=FIXTURE_IMAGES[index],
          yaw_ground_truth=0.0,
          pitch_ground_truth=0.0,
          roll_ground_truth=0.0,
        )
        for index in range(3)  # batch_size is 2, so this exercises a partial final batch
      ]
    )

    with tf.compat.v1.Session() as session:
      result = trainer._create_intermediate_image_representations(session=session, dataset=dataset)

    for datapoint in result.datapoints:
      representation_path = Path(datapoint.intermediate_representation_path)
      assert representation_path.exists()
      assert representation_path.suffix == ".npz"
      with np.load(representation_path) as data:
        assert data["arr_0"].shape == (1, mock_backbone.output_tensor_size)

    # File names are derived from the datapoint id with its extension stripped.
    assert Path(result.datapoints[0].intermediate_representation_path).name == "sample_0.npz"
