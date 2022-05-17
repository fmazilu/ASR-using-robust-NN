# This file is used for testing in development functions
import tensorflow as tf
import numpy as np
import random
import librosa
from librosa.display import specshow
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from attacks import add_noise, mixtgauss
import colorednoise as cn
from scipy.io.wavfile import write

# inputs = tf.random.normal([32, 10, 8])
# lstm = tf.keras.layers.LSTM(4)
# output = lstm(inputs)
# print(output.shape)
#
# input_shape = (4, 10, 128)
# x = tf.random.normal(input_shape)
# y = tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=input_shape[1:])(x)
# print(y.shape)
#
# input_shape = (4, 28, 28, 3)
# x = tf.random.normal(input_shape)
# y = tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=input_shape[1:])(x)
# print(y.shape)

raw_w, sampling_rate = librosa.load('C:/UPB/Licenta/GitHub/Speaker recognition/dataset/rodigits/006/006_10_0023.wav', mono=True)
raw_w = raw_w/np.max(np.abs(raw_w))
f, t, spec = spectrogram(raw_w, 16000)
plt.plot(raw_w)
plt.show()

S = librosa.feature.melspectrogram(y=raw_w, sr=sampling_rate, n_mels=128,
                                   fmax=8000)
mfccs = librosa.feature.mfcc(y=raw_w, sr=sampling_rate, n_mfcc=40)
fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                               x_axis='time', y_axis='mel', fmax=8000,
                               ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].set(title='Mel spectrogram')
ax[0].label_outer()
img = librosa.display.specshow(mfccs, x_axis='time', y_axis='mel', fmax=8000, ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='MFCC')
mfcc = librosa.feature.mfcc(raw_w)
m_htk = librosa.feature.mfcc(y=raw_w, sr=sampling_rate, dct_type=3)
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=False)
img1 = librosa.display.specshow(mfcc, x_axis='time', y_axis='mel', fmax=8000, ax=ax[0])
ax[0].set(title='RASTAMAT / Auditory toolbox (dct_type=2)')
fig.colorbar(img, ax=[ax[0]])
img2 = librosa.display.specshow(m_htk, x_axis='time', ax=ax[1])
ax[1].set(title='HTK-style (dct_type=3)')
fig.colorbar(img2, ax=[ax[1]])
# spec = spec/np.max(np.abs(spec))
# plt.pcolormesh(t, f, spec)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
plt.show()

# sr = 16000
# n_fft = 512
# n = 10
# f = np.linspace(0, sr/2, int((n_fft/2)+1))
# mels = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n)
# # mels = mels.T
# mels /= np.max(mels, axis=-1)[:, None]
# plt.plot(f, mels.T)
# plt.title("Mel-scale filter bank")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Amplitude")
# plt.show()

