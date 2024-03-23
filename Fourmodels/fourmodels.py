# Fourmodels.py

import os
import numpy as np
import librosa
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, LSTM, GRU
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def extract_features(input_folder):
    input_files = os.listdir(input_folder)
    X = []
    y = []

    for input_file_name in input_files:
        input_file_path = os.path.join(input_folder, input_file_name)

        audio, sr = librosa.load(input_file_path)

        mfccs = librosa.feature.mfcc(y=audio, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)

        features = np.concatenate([
                                    mfccs.mean(axis=1),
                                    chroma_stft.mean(axis=1),
                                    tonnetz.mean(axis=1)])

        X.append(features)

        # 파일 이름에서 클래스 추출 (예를 들어, 파일 이름이 'class1_file1.mp3'인 경우 클래스는 'class1'이 됨)
        label = input_file_name.split('_')[0]
        y.append(label)

    # 레이블을 이진 분류로 변환
    y_binary = np.where(np.array(y) == 'ai', 0, 1)
    return np.array(X), y_binary

def compare_models(input_folder, model_types):
    X, y = extract_features(input_folder)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    X_train_reshaped = np.expand_dims(X_train, axis=2)
    X_val_reshaped = np.expand_dims(X_val, axis=2)

    best_model = None
    best_accuracy = 0
    best_model_type = None

    for model_type in model_types:
        if model_type == 'CNN':
            model = Sequential()
            model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(1, activation='sigmoid')) 
            optimizer = Adam(learning_rate=0.001)
            model.compile(loss='binary_crossentropy', optimizer=optimizer ,metrics=['accuracy'])

            model.fit(X_train_reshaped, y_train, batch_size=64, epochs=100, validation_data=(X_val_reshaped, y_val), verbose=0)

            _, accuracy = model.evaluate(X_val_reshaped, y_val)

        elif model_type == 'CNN+LSTM':

            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

            batch_size = 32
            epochs = 100

            model = Sequential()

            model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.25))

            model.add(LSTM(50, return_sequences=True))
            model.add(LSTM(50))

            model.add(Flatten())

            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification (0 or 1)

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
            _, accuracy = model.evaluate(X_val, y_val)

        elif model_type == 'CNN+RNN':
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

            batch_size = 32
            epochs = 100

            model = Sequential()

            model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.25))

            model.add(GRU(50, return_sequences=True))
            model.add(GRU(50))

            model.add(Flatten())

            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification (0 or 1)

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
            _, accuracy = model.evaluate(X_val, y_val)

        elif model_type == 'XGBOOST':
            xgb_model = XGBClassifier()

            X_train_reshaped = X_train.reshape((X_train.shape[0], -1))
            X_val_reshaped = X_val.reshape((X_val.shape[0], -1))

            xgb_model.fit(X_train_reshaped, y_train)

            y_pred = xgb_model.predict(X_val_reshaped)
            accuracy = accuracy_score(y_val, y_pred)

        else:
            raise ValueError(f"Invalid model type: {model_type}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_type = model_type

    print(f"Best model type: {best_model_type}")
    return best_model, best_accuracy
