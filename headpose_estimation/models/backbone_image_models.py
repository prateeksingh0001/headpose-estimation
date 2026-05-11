from typing import List, Tuple

import tensorflow as tf

from headpose_estimation.utils.tensorflow_model_handler import TensorflowV1ModelConfig
from headpose_estimation.models.base import BasePretrainedBackBoneImageModel


class InceptionV3Model(BasePretrainedBackBoneImageModel):
    """
    Wrapper class for the InceptionV3 backbone model.
    """

    def forward(self, input_data: List[bytes]) -> tf.Tensor:
        output: List[tf.Tensor] = []
        for datapoint in input_data:
            image_representation_output = self.session.run(
                [self.image_representation_output],
                feed_dict={
                    self.pretrained_img_model_input_node: datapoint
                }
            )
            output.append(image_representation_output)
        
        return output
            

class MobileNetModel(BasePretrainedBackBoneImageModel):
    def __init__(
        self,
        tf_model_config: TensorflowV1ModelConfig,
        session = None
    ):

        self.image_resize_input_node, self.image_resize_output_node = self._get_input_preprocessing_nodes(
            pretrained_model_input_size=tf_model_config.image_input_size,
            input_image_depth=tf_model_config.image_depth
        )

        super().__init__(tf_model_config, session)

    @staticmethod
    def _get_input_preprocessing_nodes(
        pretrained_model_input_size: Tuple[int, int],
        input_image_depth: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        input_image_data = tf.placeholder(dtype=tf.float32)

        resized_image_shape = [None, *pretrained_model_input_size, input_image_depth]
        resized_image = tf.image.resize_bilinear(input_image_data, resized_image_shape)

        return input_image_data, resized_image
    
    def forward(self, input_data: tf.Tensor) -> tf.Tensor:
        return super().forward(input_data)