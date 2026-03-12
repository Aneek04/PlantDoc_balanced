# Balanced PlantDoc Dataset

This repository contains a **class-balanced version of the PlantDoc dataset**, prepared for training deep learning models for **plant disease detection and classification**.

## Overview

The original **PlantDoc dataset** consists of real-world images of plant leaves captured in natural conditions. While the dataset is valuable for building robust models, it suffers from **significant class imbalance**, where some disease classes contain many more samples than others. This imbalance can bias machine learning models toward majority classes and degrade performance on minority classes.

To address this issue, this repository provides a **balanced version of the dataset** where each class contains approximately the same number of images.

## Dataset Preparation

The dataset was balanced using **data augmentation techniques applied to minority classes**. Augmentation was performed until each class reached the size of the largest class in the original dataset.

The following augmentation methods were used:

* Random rotations
* Horizontal flipping
* Width and height shifts
* Shearing transformations
* Zoom transformations

These transformations help generate realistic variations of plant leaf images while preserving disease characteristics.

## Dataset Structure

The dataset follows a **standard image classification directory format**:

```
PlantDoc-Balanced/
    train/
        class_1/
        class_2/
        class_3/
        ...
```

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
