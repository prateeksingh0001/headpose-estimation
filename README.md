# Transfer-learning-for-head-pose-detection

This project is an twist on the [Hopenet Architecture](https://arxiv.org/pdf/1710.00925.pdf), and builds up on
this tensorflow [example script](https://github.com/tensorflow/tensorflow/blob/v1.7.0/tensorflow/examples/image_retraining/retrain.py) for
retraining image classification models.


## Datasets
1. 300W-LP synthesized large-pose face images (http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
2. UPNA Headpose estimation dataset (https://www.unavarra.es/gi4e/databases/hpdb)
3. UPNA Synthetic Head Pose Database (https://www.unavarra.es/gi4e/databases/shpdb)


## Changelogs


## Todo

### Running the code
- [x] **Download the datasets**
  - present under `./data/raw`


- [x] **Dataset extraction**
  - [x] Extract the dataset and draw the yaw, pitch and roll axis on the images
  - [x] Try using the roll, yaw and pitch values to draw axis on the face images
    - [x] setup env and run a sample
    - [x] check for the correctness of the values
  - [x] Bring the dataset to a format which can be used by the current script
    - [x] Check if it's needed in a certain format to create bottlenecks
      - The images can be left as it is but the euler angles need to be put in a `\t` detlimited csv file
      -  [x] Create the csv file from above
    - [x] Find the script for bottlenecks
      - Done by the retrain.py script


- [x] **Make the code run**
  - [x] Take the code through the debugger make changes where neccessary and get the code to a correct running state


### Refactoring the code
- [ ] Formulate a refactoring strategy
  - [] Refactor the cmdline argument parsing and passing to the main function.
  - [] Formulate a class structure which reduces cross object variable access and function calling.(more modular)
  - [] Introduce typing annotations


#### Author: Prateek Singh