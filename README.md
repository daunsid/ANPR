# Automatic Number Plate Recognition (ANPR)

This module implements an ANPR using deep learning models for both object detection and text recognition

## Datasets
The dataset is the ANPR dataset publicly available on kaggle https://www.kaggle.com/datasets/norbertelter/anpr-dataset

## Getting started
Create a virtual environment
```
    python3 -m venv venv
    source venv/bin/activate
```
Install all the necessary libraries
```
    pip3 install -r requirements.txt
```

## Quick Start
```
from main import recognize_license_plate

# results = recognize_license_plate(image_path)
print(results)
```