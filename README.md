# Transfer learning for Head Pose Estimation

This project implements a system for training headpose estimation models using transfer learning on Inception V3 and Mobilenet models.
Originally inspired by this tensorflow [example](https://github.com/tensorflow/tensorflow/blob/v1.7.0/tensorflow/examples/image_retraining/retrain.py).


## Setup
- Create a conda or venv environment with Python 3.7
- Install the dependencies using `uv pip install ".[test]"`
- Run the unit tests with `pytest tests`


## Datasets
1. 300W-LP synthesized large-pose face images (http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
2. UPNA Headpose estimation dataset (https://www.unavarra.es/gi4e/databases/hpdb)
3. UPNA Synthetic Head Pose Database (https://www.unavarra.es/gi4e/databases/shpdb)


## Contribution Guide
- Install the dev dependencies using `uv pip install ".[dev]"`
- Install pre-commit hooks with `pre-commit install`


## Potential Extensions
- Add the [Hopenet Architecture](https://arxiv.org/pdf/1710.00925.pdf), by introducing a new angle prediction head. 


#### Author: Prateek Singh
