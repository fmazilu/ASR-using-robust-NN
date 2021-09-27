import os
import pathlib
import librosa
import pandas as pd
import numpy as np
import csv
import random
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tensorflow.keras.utils import to_categorical
from IPython import display
from sklearn.utils import shuffle


#
# EXTRACT MFCC FEATURES
#
def extract_features(file_path, utterance_length):
    # Get raw .wav data and sampling rate from librosa's load function
    raw_w, sampling_rate = librosa.load(file_path, mono=True)

    # Obtain MFCC Features from raw data
    mfcc_features = librosa.feature.mfcc(raw_w, sampling_rate)
    if mfcc_features.shape[1] > utterance_length:
        mfcc_features = mfcc_features[:, 0:utterance_length]
    else:
        mfcc_features = np.pad(mfcc_features, ((0, 0), (0, utterance_length - mfcc_features.shape[1])),
                               mode='constant', constant_values=0)

    return mfcc_features


#
# DISPLAY FEATURE SHAPE
#
# wav_file_path: Input a file path to a .wav file
# Not used
def display_power_spectrum(wav_file_path, utterance_length):
    mfcc = extract_features(wav_file_path, utterance_length)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.show()

    # Feature information
    print("Feature Shape: ", mfcc.shape)
    print("Features: ", mfcc[:, 0])


# Not used
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


# Returns labels
## Not used
def get_label(file):
    parts = tf.strings.split(file, os.path.sep)

    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]


# Returns waveform and label
## Not used
def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


# Returns spectrogram for one recording
## Not used
def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
         equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram


# Plots spectrogram
## Not used
def plot_spectrogram(spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns).
    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

# Returns file names and labels of dataset
def get_file_names_and_labels(file_path):
### Gets dataset
    digit = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    data_dir = file_path
    digit_commands = []
    filenames = []
    labels = []
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    for x in digit:
        if x in commands:
            digit_commands = np.append(digit_commands, x)
    print('Commands:', digit_commands)

    i = 0
    for x in digit_commands:
        data_dir1 = str(data_dir) + '\\' + str(x)
        print(data_dir1)
        filenames += tf.io.gfile.glob(str(data_dir1) + '/*')
        for index in range(len(os.listdir(data_dir1))):
            labels.append(i)
        i = i + 1

    labels = np.array(labels)

    return filenames, labels


# Compute MFCC for all files in dataset
def compute_mfcc_all_files(filenames):
    mfcc_whole_dataset = np.zeros((len(filenames), 20, 44))
    mfcc_whole_dataset_flattened = np.zeros((len(filenames), 20 * 44))
    for index in range(len(filenames)):
        mfcc_whole_dataset[index] = extract_features(filenames[index], 44)
        mfcc_whole_dataset_flattened[index] = mfcc_whole_dataset[index].flatten()
    return mfcc_whole_dataset_flattened


# returns norms of weights in unconstrained model
def get_norms(model):
    norms = []
    for x in model.layers:
        if 'dense' in x.name:
            w = x.get_weights()[0]
            norm = np.linalg.norm(w, ord=2)
            norms = np.append(norms, norm)
    return norms

# Calculate upper-bound Lipschitz constant for unconstrained model
def get_upper_lipschitz(norms):
    return np.prod(norms)


def get_lipschitz_constrained(model):
    cst = []
    w_list = []
    for x in model.layers:
        if 'dense' in x.name:
            w = x.get_weights()[0]
            w_list.append(w)

    for index in reversed(range(len(w_list))):
        if cst == []:
            cst = np.array(w_list[index]).transpose()
        else:
            cst = np.matmul(cst, np.array(w_list[index]).transpose())

    cst = np.linalg.norm(cst, ord=2)
    return cst



if __name__ == '__main__':

    data_dir = pathlib.Path('data\\')
    save_dir = 'processed_google_dataset'
    # Get files
    filenames, labels = get_file_names_and_labels(data_dir)

    #shuffle files and labels at the same time
    filenames, labels = shuffle(filenames, labels)

    # Compute MFCC for all files
    mfcc_whole_dataset_flattened = compute_mfcc_all_files(filenames)

    # save obtained coefficients in train/dev/test datasets
    mfcc_train = mfcc_whole_dataset_flattened[:int(int(mfcc_whole_dataset_flattened.shape[0])*0.7)]
    mfcc_dev = mfcc_whole_dataset_flattened[int(int(mfcc_whole_dataset_flattened.shape[0])*0.7): int(int(mfcc_whole_dataset_flattened.shape[0])*0.9)]
    mfcc_test = mfcc_whole_dataset_flattened[-int(int(mfcc_whole_dataset_flattened.shape[0])*0.1):]
    labels_train = labels[:int(int(labels.shape[0])*0.7)]
    labels_dev = labels[int(int(labels.shape[0])*0.7): int(int(labels.shape[0])*0.9)]
    labels_test = labels[-int(int(mfcc_whole_dataset_flattened.shape[0])*0.1):]

    np.save("processed_google_dataset\\train_data", mfcc_train)
    np.save("processed_google_dataset\\train_label", labels_train)
    np.save("processed_google_dataset\\dev_data", mfcc_dev)
    np.save("processed_google_dataset\\dev_label", labels_dev)
    np.save("processed_google_dataset\\test_data", mfcc_test)
    np.save("processed_google_dataset\\test_label", labels_test)
