from __future__ import annotations
from typing import Tuple, Dict, Union
from dataclasses import dataclass

@dataclass
class ModelConfig:
    architecture: str
    graphdef_file_path: str
    download_url: str
    input_node_name: str
    output_node_name: str
    output_tensor_size: int
    image_input_size: Tuple[int, int]
    image_depth: int
    batch_size: int = 0

    @classmethod
    def from_dict(cls, input_config: Dict[str, Union[str, int]]) -> ModelConfig:
        return cls(
            architecture = input_config["architecture"],
            graphdef_file_path = input_config["graphdef_file_path"],
            download_url = input_config["download_url"],
            input_node_name = input_config["input_node_name"],
            output_node_name = input_config["output_node_name"],
            output_tensor_size = input_config["output_tensor_size"],
            image_input_size = tuple(input_config["image_input_size"]),
            image_depth = input_config["image_depth"],
            batch_size = input_config["batch_size"]
        )

    def to_dict(self) -> Dict[str, Union[str, int]]:
        return {
            "architecture": self.architecture,
            "graphdef_file_path": self.graphdef_file_path,
            "download_url": self.download_url,
            "input_node_name": self.input_node_name,
            "output_node_name": self.output_node_name,
            "output_tensor_size": self.output_tensor_size,
            "image_input_size": list(self.image_input_size),
            "image_depth": self.image_depth,
            "batch_size": self.batch_size
        }
