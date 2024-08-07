# 25-Bird-Image-Classification

This repository contains the implementation of an image classification model using PyTorch. The notebook includes data preparation, model training, and evaluation steps for an image classification task.

## Requirements

- Python 3.10 or higher
- PyTorch 1.10 or higher
- torchvision
- tqdm
- numpy

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/RupakNeupane/25-Bird-Image-Classification.git
    cd 25-Bird-Image-Classification
    ```

2. Install the required packages:

    ```bash
    pip install torch torchvision tqdm numpy matplotlib
    ```

3. Download the dataset from Kaggle and place it in the appropriate directory. The dataset path used in the notebook is `/kaggle/input/seendatasets/Seen Datasets`.


## Overview

<!-- The notebook is structured as follows:

1. **Imports and Setup**: Importing necessary libraries and setting up configurations.
2. **Data Preparation**: Loading the dataset and applying transformations for data augmentation.
3. **Model Definition**: Defining the convolutional neural network (CNN) architecture using PyTorch.
4. **Training Loop**: Training the model with GPU acceleration, including loss computation and optimization.
5. **Evaluation**: Evaluating the model on the validation dataset and computing accuracy metrics. -->

### Data Augmentation

Data Augmentation included are:
1. RandomHoriontalFlip
2. RandomRotation
3. ColorJitter
4. RandomErasing

### Data Preparation

Data preprocessing steps include:
1. Providing paths for train and validation sets
2. Calculating mean and standard deviation for normalization
3. Applying image transformations (resize, normalize)

### Model Building

The model is built using PyTorch and includes:
- A base class for image classification
- Convolutional and Depthwise Separable Convolutional blocks
- Fully connected layers with dropout

The `ResNet9` class defines the complete model architecture.

### Model Architecture

The ResNet9 model consists of the following components:

1. Initial Convolutional Block
2. Depthwise Separable Convolutional Layers
3. Residual Blocks
4. Global Average Pooling
5. Classifier

#### Layer Details

1. **conv1**: Initial Convolutional Block
   - Conv2d: 3 input channels, 64 output channels, 3x3 kernel, stride 2, padding 1
   - BatchNorm2d: Normalizes the 64 channels
   - ReLU6: Activation function (max(0, min(x, 6)))

2. **conv2**: Depthwise Separable Convolution
   - Depthwise Conv2d: 64 channels, 3x3 kernel, stride 2, padding 1, groups=64
   - Pointwise Conv2d: 64 input channels, 128 output channels, 1x1 kernel
   - BatchNorm2d: Normalizes the 128 channels
   - ReLU6: Activation function

3. **res1**: Residual Block 1
   - Two identical Depthwise Separable Convolutions:
     - Depthwise Conv2d: 128 channels, 3x3 kernel, stride 1, padding 1, groups=128
     - Pointwise Conv2d: 128 input and output channels, 1x1 kernel
     - BatchNorm2d: Normalizes the 128 channels
     - ReLU6: Activation function

4. **conv3**: Depthwise Separable Convolution
   - Depthwise Conv2d: 128 channels, 3x3 kernel, stride 2, padding 1, groups=128
   - Pointwise Conv2d: 128 input channels, 256 output channels, 1x1 kernel
   - BatchNorm2d: Normalizes the 256 channels
   - ReLU6: Activation function

5. **conv4**: Depthwise Separable Convolution
   - Depthwise Conv2d: 256 channels, 3x3 kernel, stride 2, padding 1, groups=256
   - Pointwise Conv2d: 256 input channels, 512 output channels, 1x1 kernel
   - BatchNorm2d: Normalizes the 512 channels
   - ReLU6: Activation function

6. **res2**: Residual Block 2
   - Two identical Depthwise Separable Convolutions:
     - Depthwise Conv2d: 512 channels, 3x3 kernel, stride 1, padding 1, groups=512
     - Pointwise Conv2d: 512 input and output channels, 1x1 kernel
     - BatchNorm2d: Normalizes the 512 channels
     - ReLU6: Activation function

7. **conv5**: Depthwise Separable Convolution
   - Depthwise Conv2d: 512 channels, 3x3 kernel, stride 2, padding 1, groups=512
   - Pointwise Conv2d: 512 input channels, 1024 output channels, 1x1 kernel
   - BatchNorm2d: Normalizes the 1024 channels
   - ReLU6: Activation function

8. **avgpool**: Adaptive Average Pooling
   - AdaptiveAvgPool2d: Reduces spatial dimensions to 1x1

9. **classifier**: Fully Connected Layer
   - Dropout: With probability p=0.2
   - Linear: 1024 input features, 25 output features (classes)

Key Points:
- All convolutional layers (except the first) use depthwise separable convolutions for efficiency.
- Stride 2 is used in conv1, conv2, conv3, conv4, and conv5 for downsampling.
- ReLU6 is used as the activation function throughout the network.
- Two residual blocks (res1 and res2) are used to facilitate gradient flow.
- The final classifier uses dropout for regularization before the linear layer.

### Model Parameters

- Total parameters : 1,307,417
- Trainable parameters : 1,307,417

### Training Pipeline

The training process includes:
1. Evaluation function for validation
2. Training loop with:
   - Gradient clipping
   - Learning rate scheduler

### Saving the Model

The model is saved in two formats:
1. Model state dictionary
2. Scripted version using `torch.jit.script`


### Plotting Metrics

Functions are provided to plot:
1. Accuracy vs. Epochs
2. Loss vs. Epochs
3. Learning Rate

These plots help visualize the model's performance during training.


## Usage

1. Open the Jupyter notebook:

    ```bash
    jupyter notebook competition.ipynb
    ```

2. Run the notebook cells sequentially to execute the code. Ensure that the dataset path is correctly set and that a GPU is available for faster training.



## Contact

Rupak Neupane

neupanerupak7@gmail.com

