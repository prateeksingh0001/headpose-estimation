"""Shared fixtures for the headpose_estimation tests.

The key fixture here is ``mock_backbone``: a tiny, deterministic stand-in for a
frozen backbone graph. Real backbones ship as multi-hundred-MB ``.pb`` files;
committing one as a fixture would bloat the repo. Instead we synthesize a minimal
``GraphDef`` with seeded constant weights and write it into pytest's per-test
``tmp_path`` on every run -- a float image placeholder in, a
``[None, MOCK_OUTPUT_TENSOR_SIZE]`` representation out. This exercises the same
load / import / predict plumbing as a real backbone with no binary on disk.
"""

from collections import namedtuple

import numpy as np
import pytest
import tensorflow as tf

from headpose_estimation.models.pretrained_image_model import PretrainedBackBoneImageModel
from headpose_estimation.utils.tensorflow_model_handler import TensorflowV1ModelConfig

MOCK_IMAGE_INPUT_SIZE = (32, 32)
MOCK_OUTPUT_TENSOR_SIZE = 16
MOCK_WEIGHT_SEED = 0
MOCK_INPUT_NODE_NAME = "image_input:0"
MOCK_OUTPUT_NODE_NAME = "output_image_representation:0"

MockBackbone = namedtuple(
  "MockBackbone",
  ["graph_path", "image_input_size", "output_tensor_size", "build_model"],
)


def _write_mock_backbone_graph(graph_definition_path: str) -> None:
  # Weights are baked in as seeded tf.constants, so the graph needs no variable
  # initialization and the representation it produces is reproducible.
  graph = tf.Graph()
  with graph.as_default():
    image_input = tf.compat.v1.placeholder(
      tf.float32,
      shape=[None, MOCK_IMAGE_INPUT_SIZE[0], MOCK_IMAGE_INPUT_SIZE[1], 3],
      name="image_input",
    )
    rng = np.random.RandomState(MOCK_WEIGHT_SEED)
    weights = tf.constant(rng.randn(3, MOCK_OUTPUT_TENSOR_SIZE).astype(np.float32), name="weights")
    pooled = tf.reduce_mean(image_input, axis=[1, 2])
    representation = tf.matmul(pooled, weights)
    tf.identity(representation, name="output_image_representation")

  with tf.io.gfile.GFile(graph_definition_path, "wb") as graph_file:
    graph_file.write(graph.as_graph_def().SerializeToString())


def _mock_backbone_config(graph_definition_path: str, supports_batching: bool) -> TensorflowV1ModelConfig:
  return TensorflowV1ModelConfig(
    architecture="mock_backbone",
    graph_definition_path=graph_definition_path,
    download_url=None,
    input_node_name=MOCK_INPUT_NODE_NAME,
    output_node_name=MOCK_OUTPUT_NODE_NAME,
    output_tensor_size=MOCK_OUTPUT_TENSOR_SIZE,
    image_input_size=MOCK_IMAGE_INPUT_SIZE,
    image_depth=3,
    supports_batching=supports_batching,
  )


@pytest.fixture
def mock_backbone(tmp_path) -> MockBackbone:
  """Write a fresh mock backbone graph into tmp_path and hand back a builder for it.

  ``build_model(supports_batching=True)`` resets the default graph and loads the
  mock ``.pb`` into a real ``PretrainedBackBoneImageModel``.
  """
  graph_path = str(tmp_path / "mock_backbone.pb")
  _write_mock_backbone_graph(graph_path)

  def build_model(supports_batching: bool = True) -> PretrainedBackBoneImageModel:
    tf.compat.v1.reset_default_graph()
    return PretrainedBackBoneImageModel(_mock_backbone_config(graph_path, supports_batching))

  return MockBackbone(
    graph_path=graph_path,
    image_input_size=MOCK_IMAGE_INPUT_SIZE,
    output_tensor_size=MOCK_OUTPUT_TENSOR_SIZE,
    build_model=build_model,
  )
