# Flower Recognition Using Convolutional Neural Networks (CNN)

A deep learning project that classifies flower images into five categories using a CNN built with TensorFlow and Keras.

## Problem Statement
Many flower species look very similar in shape, color, and texture, and identifying them manually requires expertise and is slow and error-prone. Traditional machine learning methods depend on hand-crafted features and perform poorly when lighting, background, angle, and scale vary.

The goal of this project is to design and train a CNN that automatically classifies flower images into five categories (daisy, dandelion, roses, sunflowers, tulips) and to evaluate it using accuracy, loss curves, a confusion matrix, and a classification report.

## Dataset
- **Name:** TensorFlow Flowers dataset (`flower_photos`)
- **Images:** about 3,600 RGB images
- **Classes (5):** daisy, dandelion, roses, sunflowers, tulips
- **Split:** 80% training, 20% validation
- **Image size:** resized to 180 x 180

The dataset is downloaded automatically when the code runs.

## Model Architecture
| Layer | Purpose |
|---|---|
| Data Augmentation | Random flip, rotation, zoom, and contrast to reduce overfitting |
| Rescaling | Scales pixel values from 0-255 to 0-1 |
| 4 x (Conv2D + MaxPooling) | Filters 32, 64, 128, 256 to extract features |
| Dropout (0.3) | Prevents overfitting |
| Flatten + Dense (256) | Combines the features |
| Dense (5) | Output layer, one score per class |

- **Optimizer:** Adam
- **Loss:** Sparse Categorical Cross-Entropy
- **Callbacks:** Early Stopping, Reduce Learning Rate on Plateau

## Results
- **Validation Accuracy:** 77.66%
- **Validation Loss:** 0.6047
- **Validation images:** 734

### Classification Report
| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| daisy | 0.77 | 0.74 | 0.75 | 129 |
| dandelion | 0.83 | 0.84 | 0.83 | 176 |
| roses | 0.65 | 0.68 | 0.67 | 120 |
| sunflowers | 0.88 | 0.83 | 0.85 | 152 |
| tulips | 0.73 | 0.76 | 0.75 | 157 |
| **Overall accuracy** | | | **0.78** | 734 |

### Analysis
The model performs best on sunflowers (F1 0.85) and dandelions (F1 0.83), which have distinctive shapes and colors. Roses are the hardest class (F1 0.67) because their petal shape and color overlap with tulips and daisies. Macro and weighted averages are both about 0.77-0.78, so performance is balanced across classes.

### Sample Images
![Sample Images](sample_images.png)

### Training Curves
![Training Curves](training_curves.png)

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Sample Prediction
![Sample Prediction](sample_prediction.png)