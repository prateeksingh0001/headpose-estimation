**2022-02-19**

* Able to load the dataset in memory and draw a simple axis on the face images, need to check which values in the matrix
  correspond to yaw, pitch and roll to be able to draw the axis correctly.

* After successfully downloading the dataset and understanding how to use the datasets, I've been spending my time
  looking at the code and finding the origin.
    * This code was adapted from the tensorflow examples for [image-retraining](https://github.com/tensorflow/hub/tree/master/examples/image_retraining)
      which seems a bad choice, as that code was meant to retrain image classification models, and it would have been
      easier had I just created the architecture from scratch or used the specific parts of the script that downloaded
      the model and attached the final layer to it. Maybe that's what I'm doing more to come.

    * Seems like the reason for going down this path would have been, that, the original [hopenet code](https://github.com/natanielruiz/deep-head-pose)
      was written in pytorch and rather than porting that code to tensorflow, they decided to build upon the
      tensorflow image retraining script.

    * Read up on creating rotation matrices from euler angles. TLDR: calculate rotation matrices along all the three axis
      and multiply all of them and then multiply them to a point. This is a helpful [guide](https://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html)
      I found. For now I can just use the formulae, but let's revisit this and learn about this linear algebra behind it.
      To map the rotated axis from 3D to 2D space we use a simple orthogonal projection.


**2023-02-18**

* Downloaded the datasets
    * 300W-LP: The pose information of each image is in a `.mat` file. Which can be oaded in-memory using `scipy.io.loadmat`


* The pose of each face is given in euler angles - yaw, pitch and roll: <br />
  This needs to be converted to cartesian coordinates and projected to 2D space before it can be mapped on an image.
  After some research I found [a stackoverflow answer](https://stackoverflow.com/questions/32131337/how-to-render-3d-axes-given-pitch-roll-and-yaw)
  which might be. For projection we can use a simple [orthogonal projection](https://textbooks.math.gatech.edu/ila/projections.html)
    
