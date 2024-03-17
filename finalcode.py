import os
from pydub import AudioSegment 
import librosa
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, LSTM, GRU
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import glob
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def add_padding_to_audio(input_folder, output_folder, padding_duration): 
    # 입력 폴더의 모든 파일 목록 가져오기
    input_files = os.listdir(input_folder)
    
    # 입력 폴더의 각 파일에 대해 변환 작업 수행
    for input_file_name in input_files:
        # 입력 파일의 전체 경로 생성
        input_file_path = os.path.join(input_folder, input_file_name)
        audio = AudioSegment.from_file(input_file_path)
        padding = AudioSegment.silent(duration=padding_duration)
        
        # 패딩을 음성 파일의 앞에 추가합니다.
        output_audio = padding + audio
        output_file_name = input_file_name.split('.')[0] + "_padded." + input_file_name.split('.')[1]
        output_file_path = os.path.join(output_folder, output_file_name)
        output_audio.export(output_file_path, format="mp3")
        
        print(f"{input_file_name}에 1초 패딩 추가 및 저장 완료.")
    
    print("모든 파일 변환이 완료되었습니다.")

def extract_features(input_folder):
    # 입력 폴더의 모든 파일 목록 가져오기
    input_files = os.listdir(input_folder)
    X = []
    y = []
    
    for input_file_name in input_files:
        # 입력 파일의 전체 경로 생성
        input_file_path = os.path.join(input_folder, input_file_name)
        
        # 음성 파일 불러오기
        audio, sr = librosa.load(input_file_path)
        
        # 특징 추출
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
        chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        
        # 특징 벡터 생성
        features = np.concatenate([mel_spectrogram.mean(axis=1), 
                                    mfccs.mean(axis=1), 
                                    spectral_rolloff.mean(axis=1), 
                                    zero_crossing_rate.mean(), 
                                    chroma_stft.mean(axis=1), 
                                    spectral_contrast.mean(axis=1), 
                                    tonnetz.mean(axis=1)])
        
        X.append(features)
        
        # 파일 이름에서 클래스 추출 (예를 들어, 파일 이름이 'class1_file1.mp3'인 경우 클래스는 'class1'이 됨)
        label = input_file_name.split('_')[0]
        y.append(label)
    
    return np.array(X), np.array(y)

def compare_models(input_folder, model_types):
    # 특징 추출
    X, y = extract_features(input_folder)
    
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
            
        elif model_type == 'CNN+XGBOOST':
            # CNN과 XGBoost를 결합하는 경우
            model = Sequential()
            model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(X_train.shape[1], 1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.25))
            model.add(Flatten())  # 이 층의 출력을 특징 벡터로 사용합니다.

            # CNN 모델 컴파일 (학습은 진행하지 않습니다)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # 특징 추출 함수 정의
            def extract_features_cnn(model, X):
                features = model.predict(X)  # 모델을 통해 특징 추출
                return features

            # 훈련 데이터와 검증 데이터에 대한 특징 추출
            X_train_features = extract_features_cnn(model, X_train)
            X_val_features = extract_features_cnn(model, X_val)

            # XGBoost 모델 학습
            xgb_model = XGBClassifier()
            xgb_model.fit(X_train_features, y_train)

            # XGBoost 모델을 사용한 예측 및 성능 평가
            y_pred = xgb_model.predict(X_val_features)
            accuracy = accuracy_score(y_val, y_pred)
            
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
            accuracy = accuracy_score(y_val, y_pred)
            
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        # 현재 모델의 정확도가 최고 정확도보다 높으면, 최고 모델을 현재 모델로 업데이트합니다.
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    return best_model, best_accuracy

# 사용 예시
input_folder = "C:\\Users\\dltos\\Downloads\\padding"
output_folder = "C:\\Users\\dltos\\Downloads\\padding_with_audio"
padding_duration = 1000  # 1초를 밀리초로 표시
model_types = ['CNN', 'CNN+XGBOOST', 'CNN+LSTM', 'CNN+RNN', 'XGBOOST']

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

add_padding_to_audio(input_folder, output_folder, padding_duration)

best_model, best_accuracy = compare_models(output_folder, model_types)

# 최고 모델의 정확도를 출력합니다.
print(f"Best model accuracy: {best_accuracy}")

# 선택된 최고 모델을 사용하여 예측하거나 추가 분석을 수행할 수 있습니다.
