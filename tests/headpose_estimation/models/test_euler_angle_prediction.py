from typing import Callable
import pytest

import tensorflow as tf
import numpy as np

from headpose_estimation.models.euler_angle_prediction import EulerAnglesPredictionHead


class TestEulerAnglesPredictionHead:
    TEST_INPUT_SHAPE = 2048
    BATCH_SIZE = 3

    @pytest.fixture(scope="class")
    def get_euler_angle_prediction_head(self) -> Callable[[tf.Session], EulerAnglesPredictionHead]:
        def euler_angle_prediction_head(session: tf.Session) -> EulerAnglesPredictionHead:
            prediction_head = EulerAnglesPredictionHead(
                input_image_representation_tensor=tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, self.TEST_INPUT_SHAPE],
                    name="angle_prediction_input"
                ),
                layer_sizes=[1024, 1],
                session=session
            )

            return prediction_head

        return euler_angle_prediction_head

    def test_angle_prediction_head_end_to_end(
        self,
        get_euler_angle_prediction_head: Callable[[tf.Session], EulerAnglesPredictionHead]
    ) -> None:
        session = tf.Session()
        prediction_head = get_euler_angle_prediction_head(session=session)
        session.run(tf.global_variables_initializer())

        input = np.random.uniform(
            size=[self.BATCH_SIZE, self.TEST_INPUT_SHAPE]
        ).astype(np.float32)
        output = prediction_head.forward(input_tensor=input)

        assert isinstance(output, tuple)
        assert all([angle_prediction.shape == (self.BATCH_SIZE, 1) for angle_prediction in output])