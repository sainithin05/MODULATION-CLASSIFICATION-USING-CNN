# MODULATION-CLASSIFICATION-USING-CNN

A deep learning-based Automatic Modulation Classification system using the **RadioML 2016.10a** dataset and **PyTorch**.

## Overview

The model classifies wireless signals using their **I/Q samples** and evaluates classification accuracy at different **Signal-to-Noise Ratio (SNR)** levels.

## Model

* CNN-based architecture
* 2 convolutional layers
* Fully connected layers
* Batch normalization
* ReLU activation
* Adam optimizer
* Cross-entropy loss

## Dataset

**RadioML 2016.10a**

The project uses signals with SNR values from **-10 dB to 10 dB**.

## Training

The dataset is divided into training and testing sets using a **75:25 split**. The model is trained using PyTorch.

## Evaluation

Model accuracy is evaluated for different SNR levels and plotted as an **SNR vs Accuracy** graph.

## Technologies

* Python
* PyTorch
* NumPy
* Matplotlib
* RadioML 2016.10a

## Author

**D. Sai Nithin Sagar**
