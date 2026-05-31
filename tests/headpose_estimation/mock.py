from typing import List

import numpy as np

from headpose_estimation.models import PretrainedBackBoneImageModel
from headpose_estimation.utils.tensorflow_model_handler import TensorflowV1ModelConfig


class MockBackBoneModel(PretrainedBackBoneImageModel):
    def __init__(self, session=None):

        tf_model_config = TensorflowV1ModelConfig(
            architecture="mock_backbone_model",
            graph_definition_path="tests/headpose_estimation/fixtures/models/mock_model.pb",
            download_url=None,
            input_node_name="mock_conv_model/image_input:0",
            output_node_name="output_image_representation:0",
            output_tensor_size=1024,
            image_input_size=(224, 224),
            image_depth=3,
            supports_batching=True,
        )

        super().__init__(tf_model_config, session)

    def forward(self, input_data: List[bytes]) -> List[np.ndarray]:
        preprocessed_image_input: List[np.ndarray] = self._preprocess_raw_images(
            input_Images=input_data
        )

        batched_preprocessed_images = np.vstack(preprocessed_image_input, axis=0)
        image_representation = self.session.run(
            self.pretrained_img_model_output_node,
            feed_dict={
                self.pretrained_img_model_input_node: batched_preprocessed_images
            },
        )

        output_representation = np.vsplit(image_representation, len(input_data))
        return output_representation
