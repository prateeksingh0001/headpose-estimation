# Transfer-learning-for-head-pose-detection

This project is an implementation of the [Hopenet Architecture](https://arxiv.org/pdf/1710.00925.pdf), and builds up on
this tensorflow [example script](https://github.com/tensorflow/hub/tree/master/examples/image_retraining) for
retraining image classification models.


## Datasets
1. 300W-LP synthesized large-pose face images (http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
2. UPNA Headpose estimation dataset (https://www.unavarra.es/gi4e/databases/hpdb)
3. UPNA Synthetic Head Pose Database (https://www.unavarra.es/gi4e/databases/shpdb)


## Changelogs


## Todos

#### Be able to run the code
- [x] Download the datasets
- [ ] Dataset  extraction
  - [x] Extract the dataset and draw the yaw, pitch and roll axis on the images
  - [ ] Try using the roll, yaw and pitch values to draw axis on the face images
    - [x] setup env and run a sample
    - [ ] check for the correctness of the values
  - [ ] Bring the dataset to a format which can be used by the current script
- [ ] Run the code to see what results does it produce


#### Refactoring the code
- [ ] Formulate a refactoring strategy



#### Author: Prateek Singh