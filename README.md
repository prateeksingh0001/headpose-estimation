# Transfer learning for head pose detection

This project is an twist on the [Hopenet Architecture](https://arxiv.org/pdf/1710.00925.pdf), and builds up on
this tensorflow [example script](https://github.com/tensorflow/tensorflow/blob/v1.7.0/tensorflow/examples/image_retraining/retrain.py) for
retraining image classification models.


## Setup
- Create a conda or venv environment with Python 3.7
- To install the dependencies `uv pip install ".[test]"`
- To run unit tests `pytest tests`


## Datasets
1. 300W-LP synthesized large-pose face images (http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
2. UPNA Headpose estimation dataset (https://www.unavarra.es/gi4e/databases/hpdb)
3. UPNA Synthetic Head Pose Database (https://www.unavarra.es/gi4e/databases/shpdb)

## Contribution Guide
- Install the dev dependencies `uv pip install ".[dev]"`
- Install pre-commit hooks `pre-commit install`


#### Author: Prateek Singh
