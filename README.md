# Balanced PlantDoc Dataset

This repository contains a **class-balanced version of the PlantDoc dataset**, prepared for training deep learning models for **plant disease detection and classification**.

## Overview

The original **PlantDoc dataset** consists of real-world images of plant leaves captured in natural conditions. While the dataset is valuable for building robust models, it suffers from **significant class imbalance**, where some disease classes contain many more samples than others. This imbalance can bias machine learning models toward majority classes and degrade performance on minority classes.

To address this issue, this repository provides a **balanced version of the dataset** where each class contains approximately the same number of images.
## Original class distribution
<img width="1005" height="695" alt="image" src="https://github.com/user-attachments/assets/bd4deb8f-3325-4f38-a2d7-d90c7e9e60ef" />


## Dataset Preparation

The dataset was balanced using **data augmentation techniques applied to minority classes**. Augmentation was performed until each class reached the size of the largest class in the original dataset.

The following augmentation methods were used:

* Random rotations
* Horizontal flipping
* Width and height shifts
* Shearing transformations
* Zoom transformations

These transformations help generate realistic variations of plant leaf images while preserving disease characteristics.

#Code 
```python
augmentor = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

```
## Dataset Structure

The dataset follows a **standard image classification directory format**:

```
plantdoc_balanced
├── Apple Scab Leaf
├── Apple leaf
├── Apple rust leaf
├── Bell_pepper leaf
├── Bell_pepper leaf spot
├── Blueberry leaf
├── Cherry leaf
├── Corn Gray leaf spot
├── Corn leaf blight
├── Corn rust leaf
├── Peach leaf
├── Potato leaf early blight
├── Potato leaf late blight
├── Raspberry leaf
├── Soyabean leaf
├── Squash Powdery mildew leaf
├── Strawberry leaf
├── Tomato Early blight leaf
├── Tomato Septoria leaf spot
├── Tomato leaf
├── Tomato leaf bacterial spot
├── Tomato leaf late blight
├── Tomato leaf mosaic virus
├── Tomato leaf yellow virus
├── Tomato mold leaf
├── Tomato two spotted spider mites leaf
├── grape leaf
└── grape leaf black rot
```
## Final class distribution
Each folder or class has 192 samples
<img width="1390" height="590" alt="image" src="https://github.com/user-attachments/assets/6fda5a97-96d5-4f3e-985b-d264b8c4720e" />


Each folder represents a plant disease class and contains all corresponding images.

This structure makes the dataset directly compatible with popular frameworks such as:

* TensorFlow / Keras
* PyTorch
* FastAI

## Usage

You can directly load this dataset using standard data loaders.

Example (TensorFlow):

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    "PlantDoc-Balanced/train",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)
```

## Purpose

This balanced dataset is intended for:

* Training deep learning models for plant disease detection
* Research experiments on real-world agricultural datasets
* Benchmarking classification algorithms

Balancing the dataset helps improve model fairness and prevents bias toward dominant classes.

## Credits

Original dataset: **PlantDoc Dataset**

If you use this dataset in your research or projects, please consider citing the original PlantDoc dataset creators.

## License

This repository follows the same license and usage guidelines as the original PlantDoc dataset.
