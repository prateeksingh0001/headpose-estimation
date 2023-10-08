**2023-10-08**
* Updates:
  * Refactored the arguments to be parsed from both a YAML file and commandline
  * Added a yaml config to running experiments
  * upgraded python version to 3.7, since 3.6 and below do not have from \__future__ import annotations
  * The code works fine with the YAML config file
* Next steps:
  * Time to think about the rest of the code and how it should be organized between classes.
    * Need to come up with an architecture
    * Dataclasses for input data and the labels(would be better to ensure there is no mixup of data)
    * The last few layer(final layer to be retrained) should be detachable from the base model.
  * Move the execution script(in retrain.py) to scripts/python


**2023-10-07**

* The code finally runs end to end completely. The correctness needs to be checked. #win
* Updates:
  * Refactored the indexing code of the image ground truths
  * Refactored the code to invoke bottleneck creation, previously the images were only being resized and not passed through the model.
  * Refactored correct tensors to be called to pass the bottlenecks through the model.
  * Changed the indexing of the ground truths being passing during training. The ground truth shape is (batch_size, 3)
    * 3 because we have yaw, pitch and roll values
  * Tried tensorflow debugger, did not find it useful for debugging
    * Only allows you to set filters and stop when those filters are activated.
    * Cannot run step by step through the execution graph
    * Solutions:
      * Can look into eager execution, will need to design code such that it can be run both way through eager execution and throught compiled graph execution(low priority maybe once I'm done refactoring)
      * Can look into tool that let you load a tensorflow graph and run throught it.
  * Changed the name of the operation being converted to constant during model saving
  * Setup black file formatting.
* Next steps:
  * Parametrizing the arguments
    1. Need to create a config class that can either read the args from command line or read them from a YAML config
    2. Refactor the code to pull arguments from the config class rather than through the `FLAGS` and `UNPARSED` variables


**2023-09-24**

* Made some way into running the code, current updates:
  * The model loads successfully
  * The bottleneck creation is working fine
  * Created a subset of the dataset because, creating bottlenecks for the entire dataset filled up my disk
* Code refactors:
  * Changed the API used for globbing files from tensorflow to pathlib
  * Loading and saving numpy objects to disk. The previous code was converting each value in the array to string one-by-one(not sure if that was even working). The code now uses np.save and np.load functions
  * Added typing-extensions to a few places, in the new code
  * The bottlenecking code was calling wrong tensors and was passing inputs to the wrong tensors in the feed_dict, corrected that.
  * Fixed the code to call correct tensorflow APIs, not sure if the wrong APIs were because of the tf version change or a bug in the code from when it was originally authored.
* Next steps:
  * Complete the end-to-end execution, the code should be able to finish a training run
  * Move the configuration from command line configurations to the config file based system(YAML)
  * Come up with a logical separation of different things, currently there is a lot of back and forth between objects
  * We should use dataclasses to represent the training data records. Will make the code much more cleaner.


**2023-03-10**

* Unsuccessfully ran the code for the first time using a debugger, was able to debug some early issues in the code to
  download of the base mo*del.
* Currently the code has quite a few bugs, some of which I had to resolve as I am trying to make to code run.
* Action plan going forward:
  1. Make the code run - this might require us to refactor/clean out a few components but let's do that.
  2. For the first step we don't make any changes to the architecture, only changes and debugging to make that the
     code works.
  3. Once the code works we can work on it's architecture and clean it.


**2023-03-09**

* Changed my IDE from Intellij to VSCode, much much more lighter, with  a lot more extensions, my computer has become
  silent again. Was able to setup debugging configurations and and a complete env for development.
* Ran the code for the first time in this project, faced some errors, I'll come back to this and re-run it tomorrow.


**2023-03-05**

* Created a function under scripts/jupyter/dataset_exploration.ipynb to load image and the data(euler angles) associated
  with it using filename and overlay the axis generated from the data over the images
* Through some trial and error I was able to figure out the correct order of values to be extracted from the .mat files
* For some images it seemed that the pitch value was inconsistent with the face, even thought the face was looking up
  the pitch had a negative value.
* Currently, the ImageDataset class(which loads the image files and the headpose angles) finds images by looking for
  filenames with certain image extensions, and loads the headpose angles just as a csv file with three columns(one for
  each angle presumably). This could lead to the issue that an image is associated with the headpose angles of another
  image during training.
* Next steps in refactoring:
  * I might use a debugger here to see how the files are loaded and put in the ImageDataset, a good start to make sure
    that the data I am using is correct.
  * Good to start introducing python typing-extensions at this stage, will help with code readability.


**2023-02-19**

* Able to load the dataset in memory and draw a simple axis on the face images, need to check which values in the matrix
  correspond to yaw, pitch and roll to be able to draw the axis correctly.

* After successfully downloading the dataset and understanding how to use the datasets, I've been spending my time
  looking at the code and finding its origin.
    * This code was adapted from the tensorflow examples for [image-retraining](https://github.com/tensorflow/hub/tree/master/examples/image_retraining)
      which seems like a bad choice, as that code was meant to retrain image classification models, and it would have been
      easier had I just created the architecture from scratch or used the specific parts of the script that downloaded
      the model and attached the final layer to it. Maybe that's what I'm doing more to come.

    * Seems like the reason for going down this path would have been, that, the original [hopenet code](https://github.com/natanielruiz/deep-head-pose)
      was written in pytorch and rather than porting that code to tensorflow, they decided to build upon the
      tensorflow image retraining script. But then this architecture is different from the hopenet in that it predict
      the angles directly rather than bucketing them followed by fine-tuned angle prediction.

    * Read up on creating rotation matrices from euler angles. TLDR: calculate rotation matrices along all the three axis
      and multiply all of them and then multiply them to a point. This is a helpful [guide](https://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html)
      I found. For now I can just use the formulae, but let's revisit this and learn about this linear algebra behind it.
      To map the rotated axis from 3D to 2D space we use a simple orthogonal projection.
      UPDATE: They're doing a simple orthonormal transformation to bring it down to 2D, just ignore the z-axis


**2023-02-18**

* Downloaded the datasets
    * 300W-LP: The pose information of each image is in a `.mat` file. Which can be loaded in-memory using
      `scipy.io.loadmat`


* The pose of each face is given in euler angles - yaw, pitch and roll: <br />
  This needs to be converted to cartesian coordinates and projected to 2D space before it can be mapped on an image.
  After some research I found [a stackoverflow answer](https://stackoverflow.com/questions/32131337/how-to-render-3d-axes-given-pitch-roll-and-yaw)
  which might be. For projection we can use a simple [orthogonal projection](https://textbooks.math.gatech.edu/ila/projections.html)
    
