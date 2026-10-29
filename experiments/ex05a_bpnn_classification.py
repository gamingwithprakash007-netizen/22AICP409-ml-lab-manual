from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)

circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})
sns.scatterplot(data=circles, x="X0", y="X1", hue="label")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy']
)
model.summary()

history = model.fit(X_train, y_train, epochs=10)
model.evaluate(X_test, y_test)

# Plot Model's Loss Curve
plt.figure()
plt.title("Model's Loss")
plt.plot(history.history['loss'])
plt.xlabel("No of Epoch")
plt.ylabel("Loss Value")
plt.show()

# Decision Boundary Helper Function
def plotq(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    x_in = np.c_[xx.ravel(), yy.ravel()]
    y_pred = model.predict(x_in)
    if len(y_pred[0]) > 1:
        print("doing multiclass classification")
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary")
        y_pred = np.round(y_pred).reshape(xx.shape)
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Training')
plotq(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title('Testing')
plotq(model, X_test, y_test)
plt.show()
