import os
import numpy as np
import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
import matplotlib.pyplot as plt
import pydub
import pathlib
import tensorflow as tf


PATH = 'dataset\\rodigits'

sample_rate, audio = wavfile.read(PATH + '\\044\\044_10_0001.wav')
data_dir = pathlib.Path('data\\')
commands = np.array(tf.io.gfile.listdir(str(data_dir)))

def normalize_audio(x):
    x = x / np.max(np.abs(x))
    return x


audio = normalize_audio(audio)

plt.figure(figsize=(15, 4))
plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(audio)), audio)
plt.grid(True)
# plt.show()


def frame_audio(audio, FFT_size=2048, hop_size=20, sample_rate=44100):
    # hop_size in ms

    audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
    frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - FFT_size) / frame_len) + 1
    frames = np.zeros((frame_num,FFT_size))

    for n in range(frame_num):
        frames[n] = audio[n*frame_len:n*frame_len+FFT_size]

    return frames


hop_size = 20  # ms
FFT_size = 2048

audio_framed = frame_audio(audio, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)
print("Framed audio shape: {0}".format(audio_framed.shape))


window = get_window("hann", FFT_size, fftbins=True)
plt.figure(figsize=(15,4))
plt.plot(window)
plt.grid(True)
# plt.show()

audio_win = audio_framed * window

ind = 71
plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 1)
plt.plot(audio_framed[ind])
plt.title('Original Frame')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(audio_win[ind])
plt.title('Frame After Windowing')
plt.grid(True)
plt.show()