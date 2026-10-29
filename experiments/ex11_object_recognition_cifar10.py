import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from tensorflow import random
import joblib
import pandas as pd

# 1. Load CIFAR-10 Dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
cifar_classes = np.array([
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'frog', 'dog', 'horse', 'ship', 'truck'
])

# 2. Preview 9 Training Samples
plt.figure(figsize=(10, 10))
plt.suptitle('Examples of training images and their labels')
for i in range(9):
    plt.subplot(331 + i, title=f'{y_train[i, 0]}-{cifar_classes[y_train[i, 0]]}')
    plt.imshow(X_train[i])
plt.show()

# 3. One-Hot Encode & Normalize Pixels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

print(f"Shape of training data:\nX-train_shape={X_train.shape} , y-train_shape={y_train.shape}")
print(f"Shape of testing data:\nX-test_shape={X_test.shape} , y-test_shape={y_test.shape}")

# 4. Build CNN Model
def get_model():
    model = Sequential()
    model.add(Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(50, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(70, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model

random.set_seed(0)
model = get_model()
print(model.summary())

# 5. Train with ModelCheckpoint and EarlyStopping
checkpoint = ModelCheckpoint("./model/cifar10_epo_cnn.h5", monitor='val_accuracy', mode='max', save_best_only=True)
early_stop = EarlyStopping(monitor="val_accuracy", patience=5, mode="max")
call_backs = [early_stop, checkpoint]

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    verbose=True,
    validation_split=0.2
)
model.save("./models/cifar10_epo_cnn.h5")
joblib.dump(history.history, "./models/cifar10_epo_cnn.history")

# 6. Plot Accuracy & Loss Curves
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(history.history['accuracy'], 'C0')
ax[0].plot(history.history['val_accuracy'], 'C1')
ax[0].set_title(label="Accuracy curves", fontsize=16)
ax[0].set_xlabel('Epochs', fontsize=14)
ax[0].set_ylabel('Accuracy', fontsize=14)
ax[0].grid()

ax[1].plot(history.history['loss'], 'C0')
ax[1].plot(history.history['val_loss'], 'C1')
ax[1].set_title(label="Loss curves", fontsize=16)
ax[1].set_xlabel('Epochs', fontsize=14)
ax[1].set_ylabel('Loss', fontsize=14)
ax[1].grid()
plt.legend(['Training', 'Validation'], fontsize=12)
plt.show()

# 7. Predict on First 15 Test Images & Count Classes
m = 15
y_pred_ = model.predict(X_test[:m])
y_pred_class = y_pred_.argmax(-1)
unique, counts = np.unique(y_pred_class, return_counts=True)

plt.figure(figsize=(10, 6))
plt.suptitle('Testing images and their labels')
for i in range(15):
    plt.subplot(3, 5, i + 1, title=f'{cifar_classes[y_pred_class[i]]}')
    plt.imshow(X_test[i])
plt.show()

freq = list(zip(cifar_classes[unique], counts))
print(pd.DataFrame(freq, columns=["class name", "count"]))
