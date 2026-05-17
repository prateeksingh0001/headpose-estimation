from typing import Callable, List

import numpy as np
import pytest
import tensorflow as tf

from headpose_estimation.models import (
    EulerAnglesPredictionHead,
    PretrainedBackBoneImageModel,
)
from tests.headpose_estimation.mock import MockBackBoneModel


class TestBasePretrainedImageModel:
    @pytest.fixture(scope="class")
    def backbone_model(self) -> PretrainedBackBoneImageModel:
        sess = tf.Session()
        return MockBackBoneModel(session=sess)

    @pytest.mark.parametrize(
        "input_image_path",
        [
            [
                "tests/headpose_estimation/fixtures/images/AFW_134212_1_0.jpg",
                "tests/headpose_estimation/fixtures/images/IBUG_image_011_1_2.jpg",
            ],
            ["tests/headpose_estimation/fixtures/images/LFPW_image_test_0001_3.jpg"],
        ],
    )
    def test__preprocess_raw_images(
        self, backbone_model: PretrainedBackBoneImageModel, input_image_path: List[str]
    ) -> None:

        input_images: List[str] = []
        for image_path in input_image_path:
            with tf.gfile.FastGFile(image_path, "rb") as img_file_handler:
                input_images.append(img_file_handler.read())

        output: List[np.ndarray] = backbone_model._preprocess_raw_images(
            input_images=input_images
        )
        resized_image_size = (
            1,
            *backbone_model.model_config.image_input_size,
            backbone_model.model_config.image_depth,
        )

        assert len(output) == len(input_image_path)
        assert all([resized_img.shape == resized_image_size for resized_img in output])


class TestEulerAnglesPredictionHead:
    TEST_INPUT_SHAPE = 2048
    BATCH_SIZE = 5
    NUM_EULER_ANGLES = 3

    @pytest.fixture(scope="class")
    def get_euler_angle_prediction_head(
        self,
    ) -> Callable[[tf.Session], EulerAnglesPredictionHead]:
        def euler_angle_prediction_head(
            session: tf.Session,
        ) -> EulerAnglesPredictionHead:
            prediction_head = EulerAnglesPredictionHead(
                input_image_representation_tensor=tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, self.TEST_INPUT_SHAPE],
                    name="angle_prediction_input",
                ),
                layer_sizes=[1024, 1],
                session=session,
            )

            return prediction_head

        return euler_angle_prediction_head

    def test_forward(
        self,
        get_euler_angle_prediction_head: Callable[
            [tf.Session], EulerAnglesPredictionHead
        ],
    ) -> None:
        session = tf.Session()
        prediction_head = get_euler_angle_prediction_head(session=session)
        session.run(tf.global_variables_initializer())

        input = np.random.uniform(size=[self.BATCH_SIZE, self.TEST_INPUT_SHAPE]).astype(
            np.float32
        )
        output = prediction_head.forward(image_input_representation=input)

        assert isinstance(output, np.ndarray)
        assert output.shape == (self.BATCH_SIZE, self.NUM_EULER_ANGLES)
