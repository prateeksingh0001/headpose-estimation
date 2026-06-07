import numpy as np
import pytest
import tensorflow as tf

from headpose_estimation.models.pretrained_image_model import PretrainedBackBoneImageModel

FIXTURE_IMAGES = [
  "tests/headpose_estimation/fixtures/images/AFW_134212_1_0.jpg",
  "tests/headpose_estimation/fixtures/images/IBUG_image_011_1_2.jpg",
  "tests/headpose_estimation/fixtures/images/LFPW_image_test_0001_3.jpg",
]


class TestPretrainedBackBoneImageModel:
  @pytest.fixture
  def raw_images(self):
    return [tf.io.gfile.GFile(image_path, "rb").read() for image_path in FIXTURE_IMAGES]

  # --- construction / graph loading ---

  def test_input_placeholder_is_scalar_string(self, mock_backbone):
    model = mock_backbone.build_model()
    assert model.input_placeholder.dtype == tf.string
    assert model.input_placeholder.shape.as_list() == []

  def test_prediction_ops_exposes_representation(self, mock_backbone):
    model = mock_backbone.build_model()
    assert set(model.prediction_ops.keys()) == {PretrainedBackBoneImageModel.IMAGE_REPRESENTATION_KEY}

  def test_prediction_op_has_configured_output_size(self, mock_backbone):
    model = mock_backbone.build_model()
    representation_op = model.prediction_ops[PretrainedBackBoneImageModel.IMAGE_REPRESENTATION_KEY]
    assert representation_op.shape.as_list()[-1] == mock_backbone.output_tensor_size

  def test_load_graph_definition_parses_pb(self, mock_backbone):
    graph_def = PretrainedBackBoneImageModel._load_graph_definition(mock_backbone.graph_path)
    assert isinstance(graph_def, tf.compat.v1.GraphDef)
    assert len(graph_def.node) > 0

  def test_load_graph_definition_missing_file_raises(self, tmp_path):
    with pytest.raises(Exception):
      PretrainedBackBoneImageModel._load_graph_definition(str(tmp_path / "does_not_exist.pb"))

  # --- preprocessing ---

  def test_preprocessing_resizes_every_image_to_model_dimensions(self, mock_backbone, raw_images):
    model = mock_backbone.build_model()
    with tf.compat.v1.Session() as session:
      preprocessed = model._preprocess_raw_images(session, raw_images)

    expected_shape = (1, *mock_backbone.image_input_size, 3)
    assert len(preprocessed) == len(raw_images)
    assert all(image.shape == expected_shape for image in preprocessed)

  def test_preprocessing_normalizes_pixels_to_unit_range(self, mock_backbone, raw_images):
    model = mock_backbone.build_model()
    with tf.compat.v1.Session() as session:
      preprocessed = model._preprocess_raw_images(session, raw_images)

    # (pixel - 128) * 0.0078125 maps [0, 255] into [-1.0, ~0.992].
    for image in preprocessed:
      assert image.min() >= -1.0
      assert image.max() <= 1.0

  def test_preprocessing_rejects_non_jpeg_bytes(self, mock_backbone):
    model = mock_backbone.build_model()
    with tf.compat.v1.Session() as session:
      with pytest.raises(tf.errors.InvalidArgumentError):
        model._preprocess_raw_images(session, [b"this is not a jpeg"])

  # --- predict / batching branch ---

  def test_predict_batched_returns_one_representation_per_image(self, mock_backbone, raw_images):
    model = mock_backbone.build_model(supports_batching=True)
    with tf.compat.v1.Session() as session:
      representations = model.predict(session, raw_images)

    assert len(representations) == len(raw_images)
    assert all(rep.shape == (1, mock_backbone.output_tensor_size) for rep in representations)

  def test_predict_non_batched_returns_one_representation_per_image(self, mock_backbone, raw_images):
    model = mock_backbone.build_model(supports_batching=False)
    with tf.compat.v1.Session() as session:
      representations = model.predict(session, raw_images)

    assert len(representations) == len(raw_images)
    assert all(rep.shape == (1, mock_backbone.output_tensor_size) for rep in representations)

  def test_batched_and_non_batched_predictions_match(self, mock_backbone, raw_images):
    # The two branches reshape differently (vstack/vsplit vs a per-image loop) but
    # the deterministic mock graph must produce identical representations either way.
    batched_model = mock_backbone.build_model(supports_batching=True)
    with tf.compat.v1.Session() as session:
      batched = batched_model.predict(session, raw_images)

    non_batched_model = mock_backbone.build_model(supports_batching=False)
    with tf.compat.v1.Session() as session:
      non_batched = non_batched_model.predict(session, raw_images)

    for batched_rep, non_batched_rep in zip(batched, non_batched):
      assert np.allclose(batched_rep, non_batched_rep)

  def test_predict_is_deterministic(self, mock_backbone, raw_images):
    model = mock_backbone.build_model()
    with tf.compat.v1.Session() as session:
      first = model.predict(session, raw_images)
      second = model.predict(session, raw_images)

    for first_rep, second_rep in zip(first, second):
      assert np.array_equal(first_rep, second_rep)
