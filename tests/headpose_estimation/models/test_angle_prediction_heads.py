import copy

import numpy as np
import pytest
import tensorflow as tf

from headpose_estimation.models.angle_prediction_heads import (
  EulerAnglesPredictionHead,
  HopeNetPredictionHead,
)
from headpose_estimation.utils.constants import (
  PITCH_ANGLE_KEY,
  ROLL_ANGLE_KEY,
  TOTAL,
  YAW_ANGLE_KEY,
)


class BaseAnglePredictionHeadTests:
  """Construction and wiring tests shared by every angle-prediction head.

  Subclasses set ``HEAD_CLASS`` / ``FINAL_LAYER_SIZE`` / ``EXPECTED_GROUND_TRUTH_KEYS``
  and inherit the assertions defined here. The class name is deliberately not
  prefixed with ``Test`` so pytest does not collect it on its own.
  """

  INPUT_SIZE = 16
  LAYER_SIZES = [8, 4]
  BATCH_SIZE = 4

  HEAD_CLASS = None
  FINAL_LAYER_SIZE = None
  EXPECTED_GROUND_TRUTH_KEYS = set()

  @property
  def _layers_per_head(self) -> int:
    # one Dense layer per configured size, plus the final output layer
    return len(self.LAYER_SIZES) + 1

  @pytest.fixture
  def head(self):
    tf.compat.v1.reset_default_graph()
    # Deterministic weight initialization: with an all-ones training input a head
    # whose hidden units all init negative is fully dead (ReLU zeroes every
    # gradient) and would not update, making test_training_updates_every_head flaky.
    tf.compat.v1.set_random_seed(0)
    global_step = tf.compat.v1.train.create_global_step()
    learning_rate = tf.constant(0.01)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    return self.HEAD_CLASS(
      input_representation_size=self.INPUT_SIZE,
      layer_sizes=self.LAYER_SIZES,
      optimizer=optimizer,
      learning_rate_tensor=learning_rate,
      global_step=global_step,
    )

  @pytest.fixture
  def session(self, head):
    with tf.compat.v1.Session() as session:
      session.run(tf.compat.v1.global_variables_initializer())
      yield session

  def _representations(self):
    return [np.ones((1, self.INPUT_SIZE), dtype=np.float32) for _ in range(self.BATCH_SIZE)]

  def _ground_truths(self):
    return {
      YAW_ANGLE_KEY: [10.0, 20.0, 30.0, 40.0],
      PITCH_ANGLE_KEY: [5.0, 10.0, 15.0, 20.0],
      ROLL_ANGLE_KEY: [1.0, 2.0, 3.0, 4.0],
    }

  def _trainable_variables_per_head(self):
    # tf.layers.Dense creates variables in construction order: all of yaw's
    # layers, then pitch's, then roll's. Slice that flat list back into heads.
    trainable_variables = tf.compat.v1.trainable_variables()
    return [
      trainable_variables[index * self._layers_per_head : (index + 1) * self._layers_per_head] for index in range(3)
    ]

  # --- wiring ---

  def test_training_updates_every_head(self, head, session):
    trainable_variables = tf.compat.v1.trainable_variables()
    before = session.run(trainable_variables)

    head.train(session, self._representations(), self._ground_truths())

    after = session.run(trainable_variables)
    changed = [bool(np.any(b != a)) for b, a in zip(before, after)]

    for head_index in range(3):
      head_changed = changed[head_index * self._layers_per_head : (head_index + 1) * self._layers_per_head]
      assert any(head_changed), f"head {head_index} did not update during training"

  # --- construction ---

  def test_trainable_variable_count(self, head):
    assert len(tf.compat.v1.trainable_variables()) == 3 * self._layers_per_head

  def test_layers_have_no_bias(self, head):
    assert all("bias" not in variable.name for variable in tf.compat.v1.trainable_variables())

  def test_weight_matrix_shapes(self, head):
    dimensions = [self.INPUT_SIZE, *self.LAYER_SIZES, self.FINAL_LAYER_SIZE]
    expected_shapes = [[dimensions[i], dimensions[i + 1]] for i in range(len(dimensions) - 1)]
    for head_variables in self._trainable_variables_per_head():
      assert [variable.shape.as_list() for variable in head_variables] == expected_shapes

  def test_prediction_ops_keys(self, head):
    assert set(head.prediction_ops.keys()) == {YAW_ANGLE_KEY, PITCH_ANGLE_KEY, ROLL_ANGLE_KEY}

  def test_prediction_op_shapes(self, head):
    for prediction_op in head.prediction_ops.values():
      assert prediction_op.shape.as_list() == [None, 1]

  def test_loss_ops_keys(self, head):
    assert set(head.loss_ops.keys()) == {TOTAL, YAW_ANGLE_KEY, PITCH_ANGLE_KEY, ROLL_ANGLE_KEY}

  def test_training_ops_present(self, head):
    assert "optimizer_node" in head.training_ops

  def test_summary_ops_present(self, head):
    assert len(head.summary_ops) == 1

  def test_input_placeholder_shape(self, head):
    assert head.input_placeholder.shape.as_list() == [None, self.INPUT_SIZE]

  def test_ground_truth_placeholder_keys(self, head):
    assert set(head.ground_truth_placeholders.keys()) == self.EXPECTED_GROUND_TRUTH_KEYS


class TestEulerAnglesPredictionHead(BaseAnglePredictionHeadTests):
  HEAD_CLASS = EulerAnglesPredictionHead
  FINAL_LAYER_SIZE = 1
  EXPECTED_GROUND_TRUTH_KEYS = {YAW_ANGLE_KEY, PITCH_ANGLE_KEY, ROLL_ANGLE_KEY}

  def test_train_does_not_mutate_caller_ground_truths(self, head, session):
    ground_truths = self._ground_truths()
    original = copy.deepcopy(ground_truths)

    head.train(session, self._representations(), ground_truths)

    assert ground_truths == original


class TestHopeNetPredictionHead(BaseAnglePredictionHeadTests):
  HEAD_CLASS = HopeNetPredictionHead
  FINAL_LAYER_SIZE = HopeNetPredictionHead.NUM_ANGLE_BUCKETS
  EXPECTED_GROUND_TRUTH_KEYS = {
    YAW_ANGLE_KEY,
    PITCH_ANGLE_KEY,
    ROLL_ANGLE_KEY,
    HopeNetPredictionHead.YAW_CLASS_KEY,
    HopeNetPredictionHead.PITCH_CLASS_KEY,
    HopeNetPredictionHead.ROLL_CLASS_KEY,
  }

  def test_predictions_stay_within_valid_angle_range(self, head, session):
    # The angle is a softmax-weighted sum over NUM_ANGLE_BUCKETS, so it is
    # mathematically bounded to [-MAX_EULER_ANGLE, MAX_EULER_ANGLE] for any input.
    rng = np.random.RandomState(0)
    representations = (rng.randn(8, self.INPUT_SIZE) * 50).astype(np.float32)
    predictions = session.run(
      list(head.prediction_ops.values()),
      feed_dict={head.input_placeholder: representations},
    )

    limit = HopeNetPredictionHead.MAX_EULER_ANGLE
    for prediction in predictions:
      assert np.all(prediction >= -limit - 1e-4)
      assert np.all(prediction <= limit + 1e-4)

  def test_train_overflows_for_angle_beyond_bucket_range(self, head, session):
    # train() bins angles with floor((angle + MAX_EULER_ANGLE) / ANGLE_PER_BUCKET)
    # and indexes a NUM_ANGLE_BUCKETS one-hot. Angles past the supported range are
    # not clamped, so the bin index overflows the one-hot and raises.
    out_of_range_angle = HopeNetPredictionHead.MAX_EULER_ANGLE + HopeNetPredictionHead.ANGLE_PER_BUCKET + 1
    ground_truths = {
      YAW_ANGLE_KEY: [out_of_range_angle],
      PITCH_ANGLE_KEY: [0.0],
      ROLL_ANGLE_KEY: [0.0],
    }
    with pytest.raises(IndexError):
      head.train(session, [np.ones((1, self.INPUT_SIZE), dtype=np.float32)], ground_truths)
