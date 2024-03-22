#구글 코랩에서 실행
#rnnoise이용해서 노이즈 제거하기.

!git clone https://gitlab.xiph.org/xiph/rnnoise.git
!pip install soundfile pydub
!apt install -y ffmpeg
!sudo apt-get install -y autoconf automake libtool gcc

%cd rnnoise
!./autogen.sh
!./configure
!make

  
import os
import numpy as np
import librosa
import soundfile as sf
import subprocess
from pydub import AudioSegment

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

# 예시 사용
input_folder = "/content/before"
output_folder = "/content/after"
for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):
        input_path = os.path.join(input_folder, filename)
        output_filename = "ai_" + filename
        output_path = os.path.join(output_folder, output_filename)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        process_and_save(input_path, output_path, top_db=20)
