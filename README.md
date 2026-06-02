# Transfer learning for Head Pose Estimation

This project implements a system for training headpose estimation models using transfer learning on a frozen backbone image models(currently supports Inception V3 and MobileNet).

Originally inspired by this tensorflow [example](https://github.com/tensorflow/tensorflow/blob/v1.7.0/tensorflow/examples/image_retraining/retrain.py).

Also implements the angle prediction head in the HopeNet paper([https://arxiv.org/pdf/1710.00925](https://arxiv.org/pdf/1710.00925))


## Setup
- Create a conda or venv environment with Python 3.7
- Install the dependencies using `uv pip install ".[test]"`
- Run the unit tests with `pytest tests`


## Running Experiments
### Donwload the datasets
1. 300W-LP synthesized large-pose face images (http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
2. UPNA Headpose estimation dataset (https://www.unavarra.es/gi4e/databases/hpdb)
3. UPNA Synthetic Head Pose Database (https://www.unavarra.es/gi4e/databases/shpdb)

### Data Preprocessing
- `python scripts/python/run_dataset_preprocessing.py -c configs/experiments/preprocessing_config.yml`

### Training a angle prediction head
- `python scripts/python/train_model.py -c configs/experiments/test_experiment.yml`


#### Author: Prateek Singh
