import pyaudio
import numpy as np
import time
from scipy.io.wavfile import write
import os
import sys

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))

from featuresExtraction import FeaturesExtraction
from dtw import VowelIdentifier
import json

# Setup untuk pyaudio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Sampling rate
CHUNK = 1024  # Ukuran buffer
RECORD_SECONDS = 3  # Durasi rekaman setiap iterasi (dalam detik)

# Inisialisasi VowelIdentifier dengan template rata-rata
vowels = ['a', 'e', 'i', 'o', 'u']
vi_mean = VowelIdentifier()

with open('src/features/mean_features.json', 'r') as f:
    mean_feat_serialized = json.load(f)

mean_feat = {key: np.array(value) for key, value in mean_feat_serialized.items()}

for vowel in vowels:
    vi_mean.add_reference(vowel, mean_feat[vowel])

# Fungsi untuk merekam audio dan mengidentifikasi vowel
def identify_realtime_vowel():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

    print("Mulai merekam. Tekan Ctrl+C untuk menghentikan.")

    try:
        while True:
            print("Rekaman dimulai.")
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(np.frombuffer(data, dtype=np.int16))
            print("Rekaman selesai.")

            audio_data = np.hstack(frames)
            # Simpan sementara audio ke file WAV (opsional, jika diperlukan untuk debugging)
            write("temp_audio.wav", RATE, audio_data)

            # Ekstraksi fitur dari audio
            feature = FeaturesExtraction(path="", audio_file="temp_audio.wav", target=None)
            feature_mean = feature.mfccs.get_mean_features()

            # Identifikasi vowel menggunakan VowelIdentifier
            identified_vowel, distance = vi_mean.identify(feature_mean)
            print(f"Identified Vowel: {identified_vowel}, Distance: {distance}")

    except KeyboardInterrupt:
        print("Rekaman berhenti.")
        stream.stop_stream()
        stream.close()
        p.terminate()

# Jalankan fungsi identifikasi vowel secara realtime
if __name__ == "__main__":
    identify_realtime_vowel()
