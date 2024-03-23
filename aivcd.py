import os
import subprocess
from pydub import AudioSegment
import librosa
import Fourmodels

# Silence trimming function using librosa
def trim_silence(audio_path, top_db=20):
    y, sr = librosa.load(audio_path)
    yt, index = librosa.effects.trim(y, top_db=top_db)
    yt_audio_segment = AudioSegment(
        yt.tobytes(),
        frame_rate=sr,
        sample_width=y.dtype.itemsize,
        channels=1
    )
    return yt_audio_segment

# RNNoise를 사용한 노이즈 제거 함수
def denoise_audio_with_rnnoise(audio_segment):
    temp_input_path = "input_temp.raw"
    temp_output_path = "output_temp.raw"

    # AudioSegment 객체에서 raw PCM 데이터를 얻고 파일로 저장
    with open(temp_input_path, 'wb') as f:
        f.write(audio_segment.raw_data)

    # rnnoise_demo 실행
    subprocess.run(['/content/rnnoise/examples/rnnoise_demo', temp_input_path, temp_output_path])

    # RAW PCM 파일을 다시 읽어서 AudioSegment 객체로 변환
    with open(temp_output_path, 'rb') as f:
        denoised_data = f.read()
    denoised_audio_segment = AudioSegment(
        denoised_data,
        frame_rate=48000,  # rnnoise_demo는 기본적으로 48kHz 샘플레이트를 사용합니다
        sample_width=2,  # 16-bit PCM, 따라서 바이트당 샘플 너비는 2입니다.
        channels=1  # 모노 채널
    )

    # 임시 파일 제거
    os.remove(temp_input_path)
    os.remove(temp_output_path)

    return denoised_audio_segment

# 파일 처리 및 저장 함수
def process_and_save(input_path, output_path, top_db=20):
    trimmed_audio = trim_silence(input_path, top_db=top_db)
    cleaned_audio = denoise_audio_with_rnnoise(trimmed_audio)
    cleaned_audio.export(output_path, format="wav")

# 전처리 후 특징 추출과 모델 비교를 수행하는 함수
def preprocess_and_compare_models(input_folder, output_folder, model_types):
    # 폴더가 없다면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 전처리 수행
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_folder, filename)
            output_filename = "ai_" + filename
            output_path = os.path.join(output_folder, output_filename)

            process_and_save(input_path, output_path, top_db=20)

    # 특징 추출과 모델 비교는 Fourmodels의 기능을 사용합니다.
    X, y = Fourmodels.extract_features(output_folder)
    best_model, best_accuracy = Fourmodels.compare_models(X, y, model_types)

    return best_model, best_accuracy

# 사용 예
input_folder = "/content/before"
output_folder = "/content/after"
model_types = ['CNN', 'CNN+LSTM', 'CNN+RNN', 'XGBOOST']

best_model, best_accuracy = preprocess_and_compare_models(input_folder, output_folder, model_types)

# 최고 모델의 정확도를 출력합니다.
print(f"Best model accuracy: {best_accuracy}")
