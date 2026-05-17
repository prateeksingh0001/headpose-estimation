from typing import List

import numpy as np
import pytest
import tensorflow as tf

from headpose_estimation.models.base import PretrainedBackBoneImageModel
from tests.headpose_estimation.models.mock import MockBackBoneModel


class TestBasePretrainedImageModel:
    @pytest.fixture(scope="class")
    def backbone_model(self) -> PretrainedBackBoneImageModel:
        sess = tf.Session()
        return MockBackBoneModel(session=sess)

    @pytest.mark.parametrize(
        "input_image_path",
        [
            [
                "tests/headpose_estimation/models/fixtures/images/AFW_134212_1_0.jpg",
                "tests/headpose_estimation/models/fixtures/images/IBUG_image_011_1_2.jpg",
            ],
            [
                "tests/headpose_estimation/models/fixtures/images/LFPW_image_test_0001_3.jpg"
            ],
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
