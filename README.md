# Transfer-learning-for-head-pose-detection

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
  - [ ] Try using the roll, yaw and pitch values and overlay on the images
  - [ ] Bring the dataset to a format which can be used by the current script
- [ ] Run the code to see what results does it produce


#### Refactoring the code
- [ ] Formulate a refactoring strategy


## Work log

**2023-02-18**

* Downloaded the datasets 
  * 300W-LP: The pose information of each image is in a `.mat` file. Which can be oaded in-memory using `scipy.io.loadmat`


  * The pose of each face is given in euler angles - yaw, pitch and roll: <br />
  This needs to be converted to cartesian coordinates and projected to 2D space before it can be mapped on an image.
  After some research I found [a stackoverflow answer](https://stackoverflow.com/questions/32131337/how-to-render-3d-axes-given-pitch-roll-and-yaw)
  which might be. For projection we can use a simple [orthogonal projection](https://textbooks.math.gatech.edu/ila/projections.html)
    



#### Author: Prateek Singh