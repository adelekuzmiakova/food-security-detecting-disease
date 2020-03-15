## Project setup

Before we get started with any project, it is a good idea to create a [virtual environment](https://www.tensorflow.org/install/pip#2.-create-a-virtual-environment-recommended) to isolate project-specific libraries from the global environment. Python virtual environment stores dependencies required by different projects in separate places by creating virtual environments for them. It solves the “Project X depends on TensorFlow version 1.x but Project Y needs 2.x version” dilemma and keeps your global site-packages directory clean and manageable. Create a new virtual environment by running the following commands:

```bash
virtualenv -p python3.7 .env        # create a virtual environment (python3.7)
source .env/bin/activate            # activate the virtual environment
pip3 install -r requirements.txt    # install dependencies
deactivate                          # exit the virtual environment 

```

requirements.txt can be found [here](https://github.com/adelekuzmiakova/food-security-detecting-disease/blob/master/requirements.txt).

## Image classification

Image classification is the core task in computer vision. The goal of an image classification task is to read an image and **assign one label from a fixed set of categories to it**. Despite its simplicity, image classification has many applications in machine learning, web development, or data science. Additionally, many other computer vision problems, such as object detection or semantic segmentation, can be reduced to image classification.

This tutorial shows how to classify images of bean plants into 3 categories:

1. **Healthy**
2. **Angular Leaf Spot disease**
3. **Bean Rust disease**

![Alt text](assets/3classes.png?raw=true "3 Classes")


In this tutorial we will follow a standard machine learning workflow:

1. Examine the data and build an input pipeline 
2. Build the classification model
3. Train the model
4. Test the model, evaluate its performance, and fine-tune hyperparameters


We start with importing necessary libraries:

```python
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
#tfds.disable_progress_bar()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
%matplotlib inline
```


```python

```
