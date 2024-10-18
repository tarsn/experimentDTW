import noisereduce as nr
import librosa
import soundfile as sf
import os

def filter_audio(input, output, amplify = True):
    """Filter the audio signal using noise reduction."""
    
    # Load the audio file
    data, sr = librosa.load(input)

    if amplify:
        data = data * 3
    
    data = nr.reduce_noise(y=data, sr=sr, stationary=True)
    sf.write(output, data, sr)
    return