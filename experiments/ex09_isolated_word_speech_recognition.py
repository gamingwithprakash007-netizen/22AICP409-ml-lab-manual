import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adadelta
from IPython.display import Audio, display

data_path = "./datasets/isolated_word_dataset"
labels = np.array(["bed", "cat", "happy"])
n_classes = labels.shape[0]
n_mfcc = 12  # no. of mfc coefficients
t = 30       # no. of time windows on which the mfc coefficients are computed
input_shape = (-1, n_mfcc, t, 1)

# 1. Extract MFCC Features from WAV Audio
def wav2mfcc(file, t, n_mfcc):
    data, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] > t:
        mfcc = mfcc[:, :t]
    if mfcc.shape[1] < t:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, t - mfcc.shape[1])))
    return mfcc

def load_speech_dataset_features(data_path, labels):
    X, y = [], []
    for i, label in enumerate(labels):
        file_name = f"{data_path}/{label}.npy"
        try:
            data = np.load(file_name)
        except FileNotFoundError:
            data = np.array([
                wav2mfcc(file, t=t, n_mfcc=n_mfcc)
                for file in librosa.util.find_files(f"{data_path}/{label}/")
            ])
            np.save(file_name, data)
        X.append(data)
        y.append(np.full((data.shape[0], 1), i))
    X = np.vstack(X).reshape(input_shape)
    y = np.vstack(y) == np.arange(n_classes)  # one-hot encoded
    return X, y

X, y = load_speech_dataset_features(data_path, labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Build CNN Model for Spectrogram/MFCC Classification
def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(n_mfcc, t, 1)))
    model.add(Conv2D(48, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(
        loss=CategoricalCrossentropy(),
        optimizer=Adadelta(0.3),
        metrics=['accuracy']
    )
    return model

tf.random.set_seed(0)
model = get_model()
print(model.summary())

# 3. Train Model
model.fit(
    X_train, y_train,
    batch_size=50,
    epochs=50,
    verbose=True,
    validation_data=(X_test, y_test)
)
tf.keras.models.save_model(model, "./models/isolated_word_speech_recognition_model.h5")

# 4. Test on Recorded Audio Files
for label in labels:
    file_name = f"{data_path}/recorded_test_audios/{label}.wav"
    test_audio, sr = librosa.load(file_name, sr=None, mono=True)
    print(f"File Name: {label}.wav")
    display(Audio(test_audio, rate=sr))
    features = wav2mfcc(file_name, t, n_mfcc).reshape(input_shape)
    print("Predicted Output :", labels[np.argmax(model.predict(features))])
    print("-" * 40)
