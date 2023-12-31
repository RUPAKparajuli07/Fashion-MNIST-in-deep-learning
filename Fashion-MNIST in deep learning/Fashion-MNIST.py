import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
 
"""## Load Data"""
 
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
 
X_train.shape, y_train.shape
 
X_test.shape, y_test.shape
 
X_train[0]
 
y_train[0]
 
class_labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
'''
0 => T-shirt/top 
1 => Trouser 
2 => Pullover 
3 => Dress 
4 => Coat 
5 => Sandal 
6 => Shirt 
7 => Sneaker 
8 => Bag 
9 => Ankle boot '''
 
plt.imshow(X_train[0], cmap ="Greys")
 
plt.figure(figsize=(16,16))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.imshow(X_train[i],cmap="Greys")
  plt.axis('off')
  plt.title(class_labels[y_train[i]]+"="+str(y_train[i]), fontsize=20)
 
  '''
0 => T-shirt/top 
1 => Trouser 
2 => Pullover 
3 => Dress 
4 => Coat 
5 => Sandal 
6 => Shirt 
7 => Sneaker 
8 => Bag 
9 => Ankle boot '''
 
"""# Feature Scalling"""
 
X_train = X_train/255
X_test = X_test/255
 
X_train[0]
 
"""## Build Neural Network"""
 
model = keras.models.Sequential([
                         keras.layers.Flatten(input_shape=[28,28]),
                         keras.layers.Dense(units=32, activation='relu'),
                         keras.layers.Dense(units=10, activation='softmax')
])
 
model.summary()
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
model.fit(X_train, y_train, epochs=1)
 
model.fit(X_train, y_train, epochs=10)
 
"""## Test and Evaluate Neural Network Model"""
 
model.evaluate(X_test,y_test)
 
y_pred = model.predict(X_test)
 
y_pred[0].round(2)
 
np.argmax(y_pred[0].round(2))
 
'''
0 => T-shirt/top 
1 => Trouser 
2 => Pullover 
3 => Dress 
4 => Coat 
5 => Sandal 
6 => Shirt 
7 => Sneaker 
8 => Bag 
9 => Ankle boot '''
 
y_test[0]
 
plt.figure(figsize=(16,16))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.imshow(X_test[i],cmap="Greys")
  plt.axis('off')
  plt.title("Actual= {} \n Predicted = {}".format(class_labels[y_test[i]], class_labels[np.argmax(y_pred[i])]))
 
"""## Confusion Matrix"""
 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, [ np.argmax(i) for i in y_pred])
 
plt.figure(figsize=(16,9))
sns.heatmap(cm, annot=True, fmt = "d")
 
"""## Classification Report"""
 
from sklearn.metrics import classification_report
cr = classification_report(y_test, [ np.argmax(i) for i in y_pred], target_names = class_labels,)
 
print(cr)
 
"""## Save Model"""
 
model.save("MNIST_classifier_nn_model.h5")
 
model = keras.models.load_model("MNIST_classifier_nn_model.h5")
 
model.predict(X_test)import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

plt.imshow(X_train[0], cmap="gray")

plt.figure(figsize=(16, 16))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.axis('off')
    plt.title(class_labels[y_train[i]], fontsize=20)

# Feature Scaling
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build Neural Network
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1)

model.fit(X_train, y_train, epochs=10)

# Test and Evaluate Neural Network Model
model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)

plt.figure(figsize=(16, 16))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_test[i], cmap="gray")
    plt.axis('off')
    plt.title("Actual = {}\nPredicted = {}".format(class_labels[y_test[i]], class_labels[np.argmax(y_pred[i])]))

# Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, [np.argmax(i) for i in y_pred])

plt.figure(figsize=(16, 9))
sns.heatmap(cm, annot=True, fmt="d")

# Classification Report
from sklearn.metrics import classification_report

cr = classification_report(y_test, [np.argmax(i) for i in y_pred], target_names=class_labels)

print(cr)

# Save Model
model.save("MNIST_classifier_nn_model.h5")

model = tf.keras.models.load_model("MNIST_classifier_nn_model.h5")

model.predict(X_test)
