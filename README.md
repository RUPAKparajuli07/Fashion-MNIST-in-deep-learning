# Fashion MNIST Neural Network Classifier

This documentation provides an overview of a simple neural network classifier for the Fashion MNIST dataset using TensorFlow and Keras. The code loads the dataset, preprocesses the data, builds the neural network model, trains the model, evaluates its performance, and saves the trained model.

## Dependencies

Before running the code, make sure you have the following dependencies installed:

- TensorFlow (tf.keras)
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn (sklearn)

## Importing Libraries

```python
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## Load Data

The Fashion MNIST dataset is loaded using TensorFlow's built-in function `tf.keras.datasets.fashion_mnist.load_data()`. The dataset contains images of 10 different fashion items.

```python
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
```

## Data Visualization

Visualize a sample image and a grid of 25 images from the training set along with their corresponding labels.

```python
plt.imshow(X_train[0], cmap="gray")
plt.figure(figsize=(16, 16))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.axis('off')
    plt.title(class_labels[y_train[i]], fontsize=20)
```

## Feature Scaling

Normalize the pixel values to a range between 0 and 1 to improve the convergence of the neural network during training.

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

## Build Neural Network

Construct a simple neural network with two layers: a Flatten layer to convert the 2D image data into a 1D vector and two Dense layers with ReLU and softmax activations, respectively.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])
```

## Model Summary and Compilation

Display the model summary and compile the model with the chosen optimizer and loss function.

```python
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

## Model Training

Train the neural network using the training data.

```python
model.fit(X_train, y_train, epochs=10)
```

## Model Evaluation

Evaluate the trained model using the test data.

```python
model.evaluate(X_test, y_test)
```

## Prediction and Visualization

Make predictions on the test data and visualize a grid of 25 test images along with their actual and predicted labels.

```python
y_pred = model.predict(X_test)

plt.figure(figsize=(16, 16))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_test[i], cmap="gray")
    plt.axis('off')
    plt.title("Actual = {}\nPredicted = {}".format(class_labels[y_test[i]], class_labels[np.argmax(y_pred[i])]))
```

## Confusion Matrix

Generate a confusion matrix to analyze the performance of the classifier.

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, [np.argmax(i) for i in y_pred])

plt.figure(figsize=(16, 9))
sns.heatmap(cm, annot=True, fmt="d")
```

## Classification Report

Display a comprehensive classification report including precision, recall, F1-score, and support for each class.

```python
from sklearn.metrics import classification_report

cr = classification_report(y_test, [np.argmax(i) for i in y_pred], target_names=class_labels)
print(cr)
```

## Save and Load Model

Save the trained model to a file and load it back.

```python
model.save("MNIST_classifier_nn_model.h5")
model = tf.keras.models.load_model("MNIST_classifier_nn_model.h5")
```

That concludes the documentation for the Fashion MNIST Neural Network Classifier. This code provides a basic example of how to load, preprocess, build, train, and evaluate a simple neural network for a multi-class classification task. You can use this code as a starting point to explore more complex neural network architectures and advanced techniques for improving model performance.
