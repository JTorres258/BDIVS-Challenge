# Parking lot detection from an image classification viewpoint

This repository contains an example code used for parking lot detection from an image classification viewpoint. That means there is no previous parking spot localization stage, but free spaces are predicted from the whole image.

## Model

The network architecture is [ResNet](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7780459).

## Running the code

### Software requirements

This code has been tested on:
- Ubuntu 14.04 with Python 3.4 and Python 3.5.
- Windows 10 with Python 3.5 and Keras 2.0.2.

Dependencies:

0. Python (3.n versions are better)
1. [Tensorflow](https://www.tensorflow.org/install/install_windows)
2. Keras, ```sudo pip install keras```
3. numpy, ```sudo pip install numpy```
4. gflags, ```sudo pip install gflags```
5. OpenCV, ```sudo pip install opencv-python```

### Data preparation

The dataset has been split into ```training```, ```validation``` and ```testing```, and it is ready to be used after downloading. Its structure is as follows:

```
user_1/
  class_1/
    frame_000000.png
    frame_000001.png
    ...
    frame_000999.png
    
  class_2/
  ...
  class_7/
  
user_2/
  ...
  
user_N/
```
In case you want to train the system with a different dataset, it is a good idea to imitate the previous folder structure.

### Training
1. Train from scratch:
```
python3 train_custom_resnet.py [flags]
```
Use ```[flags]``` to set batch size, number of epochs, dataset directories, etc. Check ```common_flags.py``` to see the description of each flag, and the default values.

Example:
```
python3 train_custom_resnet.pyy --experiment_rootdir=./models/test_1 --train_dir=../dataset/training --val_dir=../dataset/validation --img_mode=rgb

```

2. Fine-tune by loading a pre-trained model:
```
python3 train_custom_resnet.py.py --restore_model=True --experiment_rootdir=./models/test_1 --weights_fname=model_weights.h5 --img_mode=rgb --train_dir=../dataset/training
```
The pre-trained model must be in the directory you indicate in --experiment_rootdir.


### Testing
The file test.py is not finished yet. Just right now, it is prepared to work with a model with five classes (corresponfing to hand gestures). Althought, the adaptation should be almost straightforward.
```
python3 test.py [flags]
```
Example:
```
python3 test.py --experiment_rootdir=./models/test_1 --weights_fname=model_weights.h5 --test_dir=../dataset/testing --img_mode=rgb
```

*Depending on your installation, you will need to write ```python3``` or just ```python``` to run the code.
