from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union

import yaml

from headpose_estimation.utils.tensorflow_model_handler import TensorflowV1ModelConfig


def get_inception_v3_configs() -> List[TensorflowV1ModelConfig]:
    return [
        TensorflowV1ModelConfig(
            architecture="inception_v3",
            graph_definition_path="classify_image_graph_def.pb",
            download_url="http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz",
            input_node_name="DecodeJpeg/contents:0",
            output_node_name="pool_3/_reshape:0",
            output_tensor_size=2048,
            image_input_size=(229, 229),
            image_depth=3,
            batch_size=1,
        )
    ]


def get_mobilenet_v1_configs() -> List[TensorflowV1ModelConfig]:
    model_versions = ["1.0", "0.75", "0.5", "0.25"]
    model_sizes = [224, 192, 160, 128]
    quantizations = [True, False]

    output_configs = []
    for model_version in model_versions:
        for model_size in model_sizes:
            for is_quantized in quantizations:
                model_arch_name = f"mobilenet_v1_{model_version}_{str(model_size)}"

                if is_quantized:
                    model_arch_name += "_quant"

                output_configs.append(
                    TensorflowV1ModelConfig(
                        architecture=model_arch_name,
                        graph_definition_path=f"{model_arch_name}_froze.pb",
                        download_url=f"http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/{model_arch_name}.tgz",
                        input_node_name="input:0",
                        output_node_name="MobilenetV1/Predictions/Reshape:0",
                        output_tensor_size=1001,
                        image_input_size=(model_size, model_size),
                        image_depth=3,
                    )
                )

    return output_configs


def main(cli_args: Dict[str, Any]) -> None:

    config_generators: List[Callable[[], List[TensorflowV1ModelConfig]]] = [
        get_inception_v3_configs,
        get_mobilenet_v1_configs,
    ]

    model_configurations: List[Dict[str, Union[str, int]]] = []
    for config_generator in config_generators:
        for model_config in config_generator():
            output = model_config.to_dict()
            model_configurations.append(output)

    with open(cli_args["output_path"], "w") as f:
        yaml.dump(model_configurations, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output-path", type=str, help="model configs file save path"
    )
    args = parser.parse_args()

    main(vars(args))
