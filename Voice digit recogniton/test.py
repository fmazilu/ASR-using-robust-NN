# This file is used for testing in development functions
import tensorflow as tf
import numpy as np
import random
import librosa
import matplotlib.pyplot as plt
from attacks import add_noise, mixtgauss
import colorednoise as cn
from scipy.io.wavfile import write

# target_snr_db = random.randrange(5, 30, 1) #  is input to the function
# print(target_snr_db)
#
# test_filenames = np.load("test_dataset_to_add_noise\\test_filenames.npy")
# print(test_filenames[0])
# audio, sampling_rate = librosa.load(test_filenames[0], mono=True)
#
# sample = np.asanyarray(audio)
#
# signal_avg_watts = np.mean(sample ** 2)
# signal_avg_db = 10 * np.log10(signal_avg_watts)
# noise_avg_db = signal_avg_db - target_snr_db
# noise_avg_watts = 10 ** (noise_avg_db / 10)
# mean_noise = 0
# # k = np.sqrt(1 / (10 ** (target_snr_db / 10))) // dacă vrei să normezi la puterea semnalului
# k = 1
# noise_volts = k * np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(sample))
# noise_volts_power = np.mean(noise_volts ** 2)
# noise_db = 10 * np.log10(noise_volts_power)
# noisy_signal = sample + noise_volts
#
# fig, ax = plt.subplots()
# ax.plot(noisy_signal, color='r', label=f'Noisy audio with SNR = {target_snr_db}')
# ax.plot(sample, color='b', label='Raw audio')
# ax.legend()
# ax.set_title('Audio')
# ax.set_xlabel('Number of samples')
# ax.set_ylabel('Amplitude')
# plt.show()


# plot the noise
# N = 1000
# p = 0.1
# alpha = 0.1
# sigma0 = alpha
# sigma1 = 10*alpha
#
# x = mixtgauss(N, p, sigma0, sigma1)
# print(x.shape)
#
# x2 = np.random.randn(N, 1)
# # print(x2)
# plt.plot(x2, 'r', x, 'g')
# plt.show()
# fs = 16000
# N = 100000
# f = np.linspace(0.1, fs, N)
# x2 = np.random.randn(N, 1)
# beta = 1  # pink noise
# x = cn.powerlaw_psd_gaussian(beta, N)
#
# x2 = np.random.randn(N, 1)
# plt.plot(x2, 'r', x, 'g')
#
#
# c_fft = np.fft.fft(x)
# plt.figure()
# plt.plot(np.abs(c_fft))
# plt.yscale('log')
# plt.xscale('log')
# plt.show()

raw_w, sampling_rate = librosa.load('data\\seven\\0b77ee66_nohash_2.wav', mono=True)
spectro = librosa.feature.melspectrogram(y=raw_w, sr=sampling_rate, win_length=2048, hop_length=512)
mfcc = librosa.feature.mfcc(S=librosa.power_to_db(spectro), n_mfcc=8)
audio = librosa.feature.inverse.mfcc_to_audio(mfcc=mfcc, n_mels=1024)

print(audio.shape)
print(raw_w.shape)


write('test.wav', sampling_rate, audio)



