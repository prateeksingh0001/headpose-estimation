**2026-05-11**

* Updates:
  * While refactoring I've come to realize that alot of this code contains logic specific to image classification(coming from the original 2015 code from google) and if I were to do this project again I'd just implement the complete thing from scratch. #learning
  * I wonder if using inception and mobilenet models is a good choice of models, as these models are trained for classification and I'm not sure if their representations would be good enough for headpose estimation learning. #learning
    * Maybe these were the best image representation models we had in 2018(back when I originally was implementing this project).

**2026-05-09**

* Updates:
  * Reviewed the 3-class architecture diagram for the ongoing `model_class_refactor`:
    * **Class 1** – Last layer(s) (trainable head: yaw, pitch, roll outputs)
    * **Class 2** – Base model (frozen pretrained backbone)
    * **Class 3** – Composition class; owns the image representation cache (bottlenecks) and wires 1+2 together
  * Evaluated two design directions for Class 3:
    * **General**: a `ModuleList`-style class that accepts an arbitrary list of model classes/callables, with generic caching at any stage. Reusable across projects.
    * **Specific**: model wrapper (Class 1/2) stays clean, but Class 3 is project-specific and owns the bottleneck caching contract explicitly.
  * **Decision: go specific.** Key reasons:
    * TF1 graph-mode (static graph, `sess.run`, `feed_dict`) does not compose cleanly with arbitrary callables — a generic `ModuleList.run()` fights the framework rather than fits it.
    * The caching/bottleneck logic (write `.npy` once per image, feed cached arrays back to the head graph across epochs) is inherently stateful and tied to this project's training loop. Generalizing it adds complexity with no payoff.
    * New classes (`TF1ModelInference`, `EulerAnglePredictionHead`) are not yet wired into the training loop — the specific approach is the faster path to a working end-to-end refactor.
    * The model wrapper (Class 1/2) can still be a clean, self-contained unit; only the composition layer needs to be project-specific.
* Next steps:
  * Wire `TF1ModelInference` and `EulerAnglePredictionHead` into the `main_setup` training loop via the new composition class (Class 3).
  * Class 3 should own: run base model once → cache bottlenecks → loop head training N epochs against cached bottlenecks.

**2024-05-05**

* Updates:
  * Came back to this project after quite some time.
  * Could have left a better log on the next steps, coming back to this project I had to spend time and effort and understading where I left things.
  * Given different models would have different nuances eg. Inception v3 cannot take in a batched input where as mobilenet can and the inputs to both the models need to be reshaped. It makes sense to have model specific classes that deal with the peculiarities of those individual models.
    * To start with 2 classes one for inception and the other for mobilenet
    * **\[Optional\]** We can later develop a factory class that looks at the model name and calls the underlying model class based on a rule based dispatch.

**2023-10-28**

* Updates:
  * Successfully moved the model configs, and donwloading and extraction from `TF1ModelInference` to scripts, YAML config and their own classes.
    * Downloading, unpacking and initializing the model is not in the purview of `TF1ModelInference` as a result these
      operations have been moved.
    * All the information about supported models have been moved to [tf_models.yaml](./configs/tf_models/tf_models.yaml)
    * And it's downloading, unpacking and initialization can be handled by another class.
    * The `TF1ModelInference` class only gets the path to the graphdef file, names of the input and output tensors and
      the shape of the output tensor.
    * Tested model downloading and extraction and it works as expected.
  * Next steps:
    * This concludes much of the development on the base pretrained image model.
    * Now we only need to test the class to ensure
      * It can take a batch of images, where the batch is specified in the model config
      * The forward method runs as expected.
    * Next we need to develop a model for the last layers that calculate the Yaw, pitch and roll values.

**2023-10-24**

* Updates:
  * My suspicions are true, there was a reason for putting the reshaping nodes before the model.
    * Mobilenet has no input reshaping nodes it only is able to take an input of the form (x, 224, 224, 3)
    * Inception_v3 is unable to take in a batched input in the resize node, although the nodes after that can take in a batched input
* Next steps:
  * Create a function that add the following nodes
    * Resize the input image batch(this operation can take in a 4D tensor as demonstrated [here](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/resize_images))
    * Normalization
    * Pass it to the first input of the graph that can take a 4D tensor.

**2023-10-22**

* Updates:
  * Continued refactoring the pretrained image model class.
    * Tested loading a model graph in tensorflow, inspecting its operations and running and testing it with sample
      inputs. Refere to the notebook [here](./scripts/jupyter/load_read_tf_graph.ipynb) for all the steps.
    * Based on this it turns out that I don't need the steps in the function `add_jpeg_decoding` [here](./headpose_estimation/retrain.py#L419)
      as the jpeg decoding from the bytes and the input resizing is all handled by the inception_v3 model.
    * There could be the case that the mobilenet model does not handle these things and in that case we need to add
      these operations separately.
    * Also the input normalization operations are not present in the inception_v3 model and those need to be added. But
      they are better off being handled during the dataset preprocessing stage. Maybe need to implement something like
      the preprocessing functions in the pytorch dataloaders. Also unlike the input normalization being done currently
      in retrain.py, it needs to be normalized between -1 and 1.
* Next steps:
  * Look into the downloading and refactoring part of the class, if it can be reduced.
  * Check the mobilenet graph for the presence of image decoding and input resizing operations
    * If not present, we should implement them in our class and shortcircuit for the inception model.
  * Check whether the inception_v3 model can take a stack of mulitple inputs(i.e. 4-D inputs of the shape 
    [batch_size, input_height, input_width, channels]).
    * If that's not possible we might want to add our own input nodes like in points above
  * It might be a good idea to make the baseline model more general, in case we ever want to use models that are not
    inception or mobilenet.
    * An easy idea for this is is to have some predefined models(for now inception and mobilenet) with their details
      like download paths, input/output tensors etc.
    * for the rest of the models we can provide the tf graph, model name and input/output nodes that can be directly
      passed the the sess.run(). During class instantiation
    * we can check whether the model was a string for model name or is a tf.Graph object.
  * Unit tests for [pretrained_tf_img_model.py](./headpose_estimation/models/pretrained_tf_img_model.py)
    * Potential solutions
      1. Pytest
      2. Tensorflow testing modules
    * Can refer
      1. [tensorflow unit testing best practices](https://github.com/GoogleCloudPlatform/professional-services/tree/main/examples/tensorflow-unit-testing)
      2. Can be usefule [Overview of ML Pipelines](https://developers.google.com/machine-learning/testing-debugging/pipeline/overview)
      3. Maybe useful [How to Unit Test Deep Learning: Tests in TensorFlow, mocking and test coverage](https://theaisummer.com/unit-test-deep-learning/#:~:text=Unit%20tests%20in%20Python,-Before%20we%20see%20some%20more)


**2023-10-21**

* Updates:
  * Refactored the pretrained base model image class, removed unused variables combined functions into one and reduced
    information passing across functions via global variables.
  * Loaded the inception graph into memory and investigated for inputs and outputs, was able to find them on tensorboard,
    but I need to check with an acutal input.
* Next steps:
  * Check whether the input and outputs of the graph are correct, by passing an image through the model.
  * Further refactor the class for loading the pretrained image model.
    * Move the model_info details to a config file, which is loaded during runtime
    * Some functions like download and extract etc. can be combined into one larger function
      * Current implementation go between levels of abstractions (sometime is high-level code, and the other times
        it is low-level code). Can probably solve this with proper function creation in the class.
  * Find out what other operations are being called in the original script for instantiating the backbone model.
    * Possibly move those functions either pretrained_tf_img_model.py or the new canddidate.


**2023-10-18**

* Updates:
  * Moved the base(Inception) model downloading, extraction and graph creation into a separate class
* Next steps:
  * Update the class to be able to download and load the model from itself and be also able to load
    the model from disk if already present
  * Run small unit test to ensure that the system is working as expected
  * Keep structuring the classes


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
  download the base model.
* Currently the code has quite a few bugs, some of which I had to resolve as I am trying to make to code run.
* Action plan going forward:
  1. Make the code run - this might require us to refactor/clean out a few components but let's do that.
  2. For the first step we don't make any changes to the architecture, only changes and debugging to make that the
     code works.
  3. Once the code works we can work on it's architecture and clean it.


**2023-03-09**

* Changed my IDE from Intellij to VSCode, much much more lighter, with a lot more extensions, my computer has become
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
    
