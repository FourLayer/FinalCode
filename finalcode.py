import os
import soundfile as sf
import subprocess
import librosa
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, LSTM, GRU
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import Fourmodels

# 노이즈 제거 함수
def denoise_audio(input_folder, output_folder):
    # 출력 폴더 생성 (이미 존재하면 무시)
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.wav', '_denoised.wav'))

            data, samplerate = sf.read(input_path)
            # 임시 파일 경로 변경
            temp_input_path = os.path.join(output_folder, 'input_temp.raw')
            temp_output_path = os.path.join(output_folder, 'output_temp.raw')
            sf.write(temp_input_path, data, samplerate, subtype='PCM_16')

            subprocess.run(['./examples/.libs/rnnoise_demo', temp_input_path, temp_output_path])

            # 임시 출력 파일에서 읽기
            data, samplerate = sf.read(temp_output_path, channels=1, samplerate=48000, subtype='PCM_16')
            sf.write(output_path, data, samplerate)

    print("Denoising completed. The denoised audio files are saved in the folder:", output_folder)

# 특징 추출 함수
def extract_features(output_folder):
    # 입력 폴더의 모든 파일 목록 가져오기
    output_files = os.listdir(output_folder)
    X = []
    y = []
    
    for output_file_name in output_files:
        # 입력 파일의 전체 경로 생성
        output_file_path = os.path.join(output_folder, output_file_name)
        
        # 음성 파일 불러오기
        audio, sr = librosa.load(output_file_path)
        
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
        label = output_file_name.split('_')[0]
        y.append(label)
    
    return np.array(X), np.array(y)

# 전처리 및 모델 비교
def preprocess_and_compare_models(input_folder, output_folder, model_types):
    # 노이즈 제거
    denoise_audio(input_folder, output_folder)
    
    # 특징 추출
    X, y = extract_features(output_folder)
    
    # Fourmodels 패키지를 사용하여 모델 비교
    best_model, best_accuracy = Fourmodels.compare_models(X, y, model_types)
    
    return best_model, best_accuracy

# 사용 예
input_folder = '/content/before'
output_folder = '/content/after'
model_types = ['CNN', 'CNN+LSTM', 'CNN+RNN', 'XGBOOST']

best_model, best_accuracy = preprocess_and_compare_models(input_folder, output_folder, model_types)

# 최고 모델의 정확도를 출력합니다.
print(f"Best model accuracy: {best_accuracy}")
