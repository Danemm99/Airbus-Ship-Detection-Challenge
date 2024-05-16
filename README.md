# Semantic Segmentation with UNet

This repository contains the code for training and performing inference using a UNet-based model for semantic segmentation. The model is designed for the task of ship detection in aerial images.

## Table of Contents

- [Introduction](#introduction)
- [Task](#task)
- [Files](#files)
- [Requirements](#requirements)
- [Results](#results)

## Introduction

Semantic segmentation is a computer vision task that involves classifying each pixel in an image into a specific category. In this repository, we use the UNet architecture to perform semantic segmentation for ship detection in aerial images.

## Task

[Airbus Ship Detection Challenge](https://www.kaggle.com/competitions/airbus-ship-detection)

## Files

- `pytorch-eda-unet-from-scratch-finetuning.ipynb`:

1. **Data Exploration and Preprocessing**:
   - This section delves into the exploratory data analysis (EDA) aspect, offering insights into the dataset structure, distribution, and characteristics crucial for understanding the data.

2. **Data Augmentation and Processing**:
   - Defines techniques for image augmentation, aiding in expanding the dataset and enhancing model generalization.
   - Provides an initial overview of the data, likely including loading, preprocessing, and possibly normalization steps.

3. **Dataset Handling**:
   - Introduces the `AirbusDataset` class, designed specifically for managing ship detection data within the PyTorch framework. This class likely includes methods for loading images and labels, as well as any necessary transformations.

4. **Model Architectures**:
   - Presents two convolutional neural network (CNN) architectures tailored for semantic segmentation tasks: "Improved U-Net (IUNet)" and "Original U-Net." These architectures form the backbone of the segmentation model.

5. **Training and Inference**:
   - Offers a Python script for training the U-Net model, incorporating functionalities for model initialization, loss computation, optimization, and validation.
   - Provides a Python script dedicated to performing inference with the trained U-Net model, allowing for predictions on unseen data.

6. **Evaluation Metrics and Loss Functions**:
   - Implements various loss functions tailored for image segmentation tasks, crucial for training the model effectively.
   - Defines a set of functions and a class for computing evaluation metrics, aiding in assessing the model's performance and effectiveness.

7. **Utilities and Miscellaneous**:
   - Contains utility functions and configurations essential for ship detection in satellite images.
  
- `submission.csv`: CSV file containing the inference results for submission.


## Requirements

Make sure to install the necessary dependencies before running the code. You can use the provided `requirements` file:

```
pip install -r requirements
```


## Results

Check the `submission.csv` file for the model's inference results. The result we got from `pytorch-eda-unet-from-scratch-finetuning.ipynb` notebook, is the main solution for the task.

![submission](https://github.com/Danemm99/Airbus-Ship-Detection-Challenge/assets/112890351/607adfce-d67d-49e8-8e6d-d7fd9e50e91c)

