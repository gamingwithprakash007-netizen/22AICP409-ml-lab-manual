import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    Dense, Dropout, Flatten,
    BatchNormalization
)
from tensorflow.keras.utils import to_categorical
import joblib

# 1. Load and Preprocess MNIST Data
train = pd.read_csv('../MLdatasets/mnist/train.csv')
test = pd.read_csv('../MLdatasets/mnist/test.csv')

X_train = train.drop(['label'], axis=1).values
y_train = train['label'].values
X_test = test.values

X_train = X_train.reshape(-1, 28, 28, 1) / 255
X_test = X_test.reshape(-1, 28, 28, 1) / 255
y_train = to_categorical(y_train)

# 2. Visualize Sample Digits
fig, axes = plt.subplots(3, 3, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].squeeze(), cmap='binary')
    digit = y_train[i].argmax()
    ax.set(title=f"label: {digit}")
fig.tight_layout()
plt.show()

# 3. Define CNN Model Architecture
def get_model():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

tf.random.set_seed(0)
model = get_model()
print(model.summary())

# 4. Train and Save Model
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=20,
    validation_split=0.2
)
tf.keras.models.save_model(model, "models/mnist_cnn.h5")
joblib.dump(history.history, "models/mnist_cnn.history")

# 5. Plot Loss and Accuracy Curves
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss")
ax[0].legend()

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
ax[1].legend()
plt.show()

# 6. Predict on Test Images
y_pred = model.predict(X_test)
fig, axis = plt.subplots(4, 4, figsize=(8, 10))
for i, ax in enumerate(axis.flat):
    ax.imshow(X_test[i].squeeze(), cmap='binary')
    ax.set(title=f"Prediction: {y_pred[i].argmax()}")
    ax.axis("off")
fig.suptitle("Test Predictions")
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()
