# WinstarsAI_task: Semantic Segmentation with UNet

This repository contains the code for training and performing inference using a UNet-based model for semantic segmentation. The model is designed for the task of ship detection in aerial images.

## Table of Contents

- [Introduction](#introduction)
- [Files](#files)
- [Requirements](#requirements)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Introduction

Semantic segmentation is a computer vision task that involves classifying each pixel in an image into a specific category. In this repository, we use the UNet architecture to perform semantic segmentation for ship detection in aerial images.

## Files

- `data_augmentation.py`: This file defines an image augmentation module for data preprocessing.
- `data_processing`: First overview of the data.
- `dataset.py`: This file defines an AirbusDataset class, which serves as a PyTorch dataset for handling ship detection data.
- `losses.py`: This file implements several loss functions for image segmentation task.
- `metrics.py`: This file defines a set of functions and a class for computing evaluation metrics.
- `models.py`: This file defines two convolutional neural network (CNN) architectures. The architectures are named "Improved U-Net (IUNet)" and "Original U-Net".
- `submission.py`: Create 'submission.csv' file.
- `utils.py`: This file contains utility functions and configurations essential for ship detection in satellite images.
- `exploratory_data_analysys.ipynb`: This notebook focuses on the exploratory data analysis (EDA) aspect of a semantic segmentation task using the U-Net architecture.
- `train.py`: Python script for training the UNet model.
- `inference.py`: Python script for performing inference with the trained model.
- `pytorch-eda-unet-from-scratch-finetuning.ipynb`: Сombined files mentioned above in one Jupyter notebook. `You can use this notebook for testing my solution too (run on Kaggle platform)`.
- `submission.csv`: CSV file containing the inference results for submission.


## Requirements

Make sure to install the necessary dependencies before running the code. You can use the provided `requirements` file:

```
pip install -r requirements
```


## Results

Check the `submission.csv` file for the model's inference results. The result we got from `semantic-segmentation-unet-advanced-approach.ipynb` notebook, is the main solution for the task.

![image](https://github.com/geeeeenccc/WinstarsAI_task/assets/101811004/395742c0-4cc9-4488-9d83-fbd45e731609)




## Acknowledgments

This project was developed as part of the Winstars AI task. I appreciate any feedback and contributions to improve the model and its performance.
