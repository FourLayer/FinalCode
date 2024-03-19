#compare model library code 


# model_comparison.py
# usage: import model_comparison
#	 model_comparison.compare_models()

import librosa
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import glob
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def compare_models(X, y, model_types):
    # 데이터를 훈련 세트와 검증 세트로 분할합니다.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    best_model = None
    best_accuracy = 0
    
    for model_type in model_types:
        # 각 모델 타입을 훈련하고 평가합니다.
        if model_type == 'CNN':
            # CNN 모델을 정의합니다.
            model = Sequential()
			model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(X_train.shape[1], 1)))
			model.add(MaxPooling1D(pool_size=2))
			model.add(Dropout(0.25))
			model.add(Flatten())
			model.add(Dense(1, activation='sigmoid'))
			optimizer = Adam(learning_rate=0.001)
			model.compile(loss='binary_crossentropy', optimizer=optimizer ,metrics=['accuracy'])
            # CNN 모델을 학습합니다.
            model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_val, y_val), verbose=0)
            # CNN 모델의 정확도를 측정합니다.
            _, accuracy = model.evaluate(X_val, y_val)
			
        elif model_type == 'CNN+LSTM':
            # CNN과 LSTM을 결합하는 경우
            # Reshape for LSTM
			X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
			X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

			# Define batch_size and epochs
			batch_size = 32
			epochs = 100

			# Create the model
			model = Sequential()

			# Convolutional layer
			model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
			model.add(MaxPooling1D(pool_size=2))
			model.add(Dropout(0.25))

			# LSTM layer
			model.add(LSTM(50, return_sequences=True))
			model.add(LSTM(50))

			# Flatten layer
			model.add(Flatten())

			# Fully connected layers
			model.add(Dense(128, activation='relu'))
			model.add(Dropout(0.5))
			model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification (0 or 1)

			# Compile the model
			model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

			# Train the model
			model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
			_, accuracy = model.evaluate(X_val, y_val)
			
        elif model_type == 'CNN+RNN':
            # Reshape for CNN
			X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
			X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

			# Define batch_size and epochs
			batch_size = 32
			epochs = 100

			# Create the model
			model = Sequential()

			# Convolutional layer
			model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
			model.add(MaxPooling1D(pool_size=2))
			model.add(Dropout(0.25))

			# GRU layer
			model.add(GRU(50, return_sequences=True))
			model.add(GRU(50))

			# Flatten layer
			model.add(Flatten())

			# Fully connected layers
			model.add(Dense(128, activation='relu'))
			model.add(Dropout(0.5))
			model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification (0 or 1)

			# Compile the model
			model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

			# Train the model
			model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
			
			_, accuracy = model.evaluate(X_val, y_val)

        elif model_type == 'XGBOOST':
            # XGBoost 모델
			xgb_model = XGBClassifier()
			xgb_model.fit(X_train, y_train)

			y_pred = xgb_model.predict(X_val)
			print(f"XGBoost Accuracy: {accuracy_score(y_val, y_pred)}")

            accuracy = accuracy_score(y_val, y_pred)
			
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        # 현재 모델의 정확도가 최고 정확도보다 높으면, 최고 모델을 현재 모델로 업데이트합니다.
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    return best_model, best_accuracy

# 데이터가 이미 정의되어 있다고 가정합니다.
# X = ...
# y = ...

# 비교할 모델 유형을 정의합니다.
model_types = ['CNN', 'CNN+LSTM', 'CNN+RNN', 'XGBOOST']

# 모델을 비교하고 최고 정확도를 가진 모델을 가져옵니다.
best_model, best_accuracy = compare_models(X, y, model_types)

# 최고 모델의 정확도를 출력합니다.
print(f"Best model accuracy: {best_accuracy}")

# 선택된 최고 모델을 사용하여 예측하거나 추가 분석을 수행할 수 있습니다.
