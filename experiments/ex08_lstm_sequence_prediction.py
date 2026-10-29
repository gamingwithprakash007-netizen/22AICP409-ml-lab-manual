import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam, Adadelta
from sklearn.model_selection import train_test_split

# 1. Create Arithmetic Sequence
X = np.arange(1, 50)
Y = X * 15
print("Sequence:", X, Y, sep="\n")

X = X.reshape(-1, 1, 1)
Y = Y.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False)

# 2. Define LSTM Architecture
def get_model():
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer=Adadelta(learning_rate=0.03), loss='mse')
    return model

tf.random.set_seed(0)
model = get_model()
print(model.summary())

# 3. Train Model
model.fit(X_train, Y_train, epochs=2000, validation_split=0.2, batch_size=10)
tf.keras.models.save_model(model, "./models/seq_pred_model.h5")

# 4. Predict and Compare Actual vs Predicted
Y_pred = model.predict(X_test, verbose=0)
Y_diff = Y_test - Y_pred
df_results = pd.DataFrame(
    np.hstack([Y_test, Y_pred, Y_diff]),
    columns=["Actual", "Predicted", "Difference"]
)
print(df_results)
