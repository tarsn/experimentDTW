import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import librosa.display
from python_speech_features import mfcc, delta

class FeaturesExtraction:
    def __init__(self, path, audio_file, target):
        self.audioName = audio_file
        self.target = target

        file_path = os.path.join(path, audio_file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file '{audio_file}' not found at the given path '{path}'.")

        self.y, self.sr = librosa.load(file_path)

        # Instance for extracting features of 39 MFCCs
        self.mfccs = MFCCs(self.y, self.sr)

    def plot_spectrogram(self, title='Spectrogram', show_plot=True, save_plot=False, filename='spectrogram_plot.png'):
        """Plot the spectrogram of an audio signal."""
        S = librosa.feature.melspectrogram(y=self.y, sr=self.sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=self.sr)
        plt.title(title)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(format='%+2.0f dB')
        
        if save_plot:
            plt.savefig(filename)
        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_waveform(self, title='Waveform', show_plot=True, save_plot=False, filename='waveform_plot.png'):
        """Plot the waveform of an audio signal."""
        plt.figure(figsize=(10, 4))
        plt.plot(self.y)
        plt.title(title)
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        
        if save_plot:
            plt.savefig(filename)
        if show_plot:
            plt.show()
        else:
            plt.close()


class MFCCs:
    def __init__ (self, signal, sr):
        self.signal = signal
        self.sr = sr
        self.mfcc_features = None
    
    def extract_features(self, num_mfcc=13):
        """Extract 39 MFCC features (13 MFCC, 13 Delta, 13 Delta-Delta)"""
        # Ekstrak 13 koefisien MFCCs dasar
        mfcc_features = mfcc(self.signal, samplerate=self.sr, numcep=num_mfcc, nfilt=26, nfft=1200)
        
        # Ekstrak Delta (turunan pertama)
        delta_mfcc = delta(mfcc_features, 2)
        
        # Ekstrak Delta-Delta (turunan kedua)
        delta2_mfcc = delta(delta_mfcc, 2)
        
        # Gabungkan menjadi 39 dimensi (13 MFCC + 13 Delta + 13 Delta-Delta)
        combined_features = np.hstack((mfcc_features, delta_mfcc, delta2_mfcc))
        
        self.mfcc_features = combined_features
        return combined_features
    
    def get_mean_features(self):
        """Return the mean of MFCC features."""
        if self.mfcc_features is None:
            self.extract_features()
        return np.mean(self.mfcc_features, axis=0)
    
    def plot_mfcc(self, title='MFCCs', show_plot=True, save_plot=False, filename='mfcc_plot.png'):
        """Plot the MFCCs of the audio signal."""
        # Ekstraksi MFCCs menggunakan Librosa untuk plotting (13 MFCCs dasar)
        mfcc = librosa.feature.mfcc(y=self.signal, sr=self.sr, n_mfcc=13)
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis='time', sr=self.sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('MFCC Coefficients')

        if save_plot:
            plt.savefig(filename)
        if show_plot:
            plt.show()
        else:
            plt.close()
        