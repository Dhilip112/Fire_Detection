# Wildfire Detection Using Deep Learning

## Overview
This project focuses on detecting wildfires using deep learning models. It utilizes a dataset consisting of images labeled as either "fire" or "nofire" and trains a neural network to classify them accordingly.

## Dataset
- The dataset is stored in Kaggle and contains images categorized into two classes:
  - `fire`: Images containing wildfire occurrences.
  - `nofire`: Images without any fire.
- The dataset directory structure:
  ```
  /root/.cache/kagglehub/datasets/elmadafri/the-wildfire-dataset/versions/3/the_wildfire_dataset/
  ├── train/
  │   ├── fire/
  │   ├── nofire/
  ├── val/
  │   ├── fire/
  │   ├── nofire/
  ├── test/
  │   ├── fire/
  │   ├── nofire/
  ```

## Installation & Setup
To set up the environment, install the required dependencies:
```bash
pip install tensorflow numpy matplotlib opencv-python
```

Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/your-repo/wildfire-detection.git
cd wildfire-detection
```

## GPU Verification
To ensure GPU is available for faster training, the following code is used:
```python
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('GPU is Available!')
else:
    print('GPU is Unavailable!')
```

## Data Exploration
Before training, the dataset is visualized by displaying sample images from each class using `matplotlib`.

```python
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load dataset paths
train_dir = "path_to_dataset/train"
classes = os.listdir(train_dir)

# Display images from 'fire' class
plt.figure(figsize=(12, 10))
for i in range(5):
    class_path = os.path.join(train_dir, classes[0])
    img_name = os.listdir(class_path)[i]
    img_path = os.path.join(class_path, img_name)
    img = mpimg.imread(img_path)
    
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.title(f'{classes[0]}\n shape: {img.shape}')
    plt.axis('off')
plt.show()
```

Similarly, images from the "nofire" class are visualized.

## Model Architecture
The deep learning model is built using TensorFlow/Keras. It consists of:
- **Convolutional Layers** – For feature extraction.
- **Pooling Layers** – To reduce dimensionality.
- **Fully Connected Layers** – For classification.
- **Activation Functions** – `ReLU` and `Softmax`.

## Training Details
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 20
- **Validation Split**: 20%

## Evaluation Metrics
The model performance is evaluated using:
- **Accuracy**: Measures correct classifications.
- **Precision & Recall**: Measures false positives/negatives.
- **F1-Score**: Balances precision and recall.

## Usage
To train the model, run:
```bash
python train.py
```

To test the model on new images:
```python
python predict.py --image path_to_image.jpg
```

## Results
The trained model achieves an accuracy of approximately **X%** on the validation dataset. Below is a confusion matrix for evaluation:
```
| Actual \ Predicted | Fire | No Fire |
|-------------------|------|---------|
| Fire             |  XX  |    XX   |
| No Fire         |  XX  |    XX   |
```

## Future Enhancements
- Implementing real-time wildfire detection using a webcam or drone footage.
- Deploying the trained model as a web application.

## Author
Developed by Dhilip Kumar R.

