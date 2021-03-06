import os
import pathlib
import librosa
import numpy as np
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


digit = ['006', '041', '043', '044', '045', '046', '047', '048', '049', '105', '117', '118', '211', '212',
         '213', '214', '215', '260', '261', '420']


maximum = 0


#
# EXTRACT MFCC FEATURES
#
def extract_features(file_path, utterance_length):
    global maximum
    # window = 2 * 22050  # aprox 2 seconds of audio
    # Get raw .wav data and sampling rate from librosa's load function
    raw_w, sampling_rate = librosa.load(file_path, mono=True)

    # Obtain MFCC Features from raw data
    mfcc_features = librosa.feature.mfcc(y=raw_w, sr=sampling_rate)
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
# Not used
def get_label(file):
    parts = tf.strings.split(file, os.path.sep)

    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]


# Returns waveform and label
# Not used
def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


# Returns spectrogram for one recording
# Not used
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
# Not used
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
    # Gets dataset
    digit = ['006', '041', '043', '044', '045', '046', '047', '048', '049', '105', '117', '118', '211', '212',
             '213', '214', '215', '260', '261', '420']
    data_dir = file_path
    digit_commands = []
    filenames = []
    labels = []
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    for x in digit:
        if x in commands:
            digit_commands = np.append(digit_commands, x)

    i = 0
    for x in digit_commands:
        data_dir1 = str(data_dir) + '\\' + str(x)
        filenames += tf.io.gfile.glob(str(data_dir1) + '/*')
        for index in range(len(os.listdir(data_dir1))):
            labels.append(i)
        i = i + 1

    labels = np.array(labels)

    return filenames, labels


# Compute MFCC for all files in dataset
def compute_mfcc_all_files(filenames):
    mfcc_whole_dataset = np.zeros((len(filenames), 20, 500))
    mfcc_whole_dataset_flattened = np.zeros((len(filenames), 20 * 500))
    for index in range(len(filenames)):
        mfcc_whole_dataset[index] = extract_features(filenames[index], 500)
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
    batch_layers = []
    w_list = []
    correction_factors = []
    correction_factor = 1
    for x in model.layers:
        if "batch" in x.name:
            batch_layers.append(x)
        if 'dense' in x.name:
            w = x.get_weights()[0]
            w_list.append(w)
    for layer in batch_layers:
        gamma = layer.get_weights()[0]
        variance = layer.get_weights()[3]
        correction_factors.append(np.sqrt(variance)/gamma)
    if correction_factors != []:
        correction_factor = np.prod([np.max(c) for c in correction_factors])

    for index in reversed(range(len(w_list))):
        if cst == []:
            cst = np.array(w_list[index]).transpose()
        else:
            cst = np.matmul(cst, np.array(w_list[index]).transpose())

    cst = np.linalg.norm(cst, ord=2)
    cst = cst / correction_factor
    return cst


# This function loads the audio data from all the files of the dataset
# For a file the loaded audio is of shape (1, num_samples)
# The audio samples are then split into 1 second audio arrays, with the first second and last second of audio
# being discarded
# The dataset will be of shape (num_seconds, 22050), 22050 being the sampling frequency
# This function will return an array of shape (num_seconds, mfcc_features) containing the data and
# an array of shape (num_seconds, 1) containing the labels
def load_audio_dataset_and_labels(filenames, labels):
    global maximum
    split_audio = []
    mfcc_features = []
    local_labels = []
    for i, file_path in enumerate(filenames):
        # Get raw .wav data and sampling rate from librosa's load function
        raw_w, sampling_rate = librosa.load(file_path, mono=True)
        window_length = 1 * sampling_rate  # aprox 1 second of audio
        audio_length = int(len(raw_w)/window_length)
        # Cutting off first second and last second (actually more than one second at the end) of recording
        raw_w = raw_w[window_length:(audio_length - 1) * window_length]
        # update new audio length
        audio_length = int(len(raw_w) / window_length)
        # mfcc_features = np.array(np.zeros((audio_length, 20, 44)))
        for index in range(audio_length):
            # for this file append the label to the local labels variables as many times as the windows are in the audio
            # file
            local_labels.append(labels[i])
            split_audio.append(raw_w[index * window_length: (index + 1) * window_length])

    split_audio = np.array(split_audio, dtype=np.ndarray)
    for j in range(split_audio.shape[0]):
        # Obtain MFCC Features from raw data, window length = 441 for 20ms at 22kHz sampling rate
        mfcc_features.append(librosa.feature.mfcc(y=np.array(split_audio[j], dtype=float), sr=sampling_rate,
                                                  win_length=441, n_fft=441, hop_length=220))

    mfcc_features = np.array(mfcc_features, dtype=np.ndarray)

    mfcc = mfcc_features.reshape(mfcc_features.shape[0], mfcc_features.shape[1] * mfcc_features.shape[2])
    return mfcc, np.array(local_labels)


def main():
    data_dir = pathlib.Path('dataset\\rodigits\\')
    save_dir = 'processed_google_dataset'
    # Get files
    filenames, labels = get_file_names_and_labels(data_dir)

    # shuffle files and labels at the same time
    filenames, labels = shuffle(filenames, labels)

    # Divide into train, dev and test sets before calculating MFCCs
    filenames_train = filenames[:int(int(len(filenames))*0.7)]
    filenames_dev = filenames[int(int(len(filenames))*0.7): int(int(len(filenames))*0.9)]
    filenames_test = filenames[-int(int(len(filenames))*0.1):]

    labels_train = labels[:int(int(labels.shape[0]) * 0.7)]
    labels_dev = labels[int(int(labels.shape[0])*0.7): int(int(labels.shape[0])*0.9)]
    labels_test = labels[-int(int(labels.shape[0])*0.1):]

    mfcc_train, labels_train = load_audio_dataset_and_labels(filenames_train, labels_train)
    mfcc_val, labels_val = load_audio_dataset_and_labels(filenames_dev, labels_dev)
    mfcc_test, labels_test = load_audio_dataset_and_labels(filenames_test, labels_test)

    # # Saving test filenames and labels
    np.save("test_dataset_to_add_noise\\test_label", labels_test)
    np.save("test_dataset_to_add_noise\\test_filenames", filenames_test)

    np.save("RoDigits_splitV2\\train_data", mfcc_train)
    np.save("RoDigits_splitV2\\train_label", labels_train)
    np.save("RoDigits_splitV2\\dev_data", mfcc_val)
    np.save("RoDigits_splitV2\\dev_label", labels_val)
    np.save("RoDigits_splitV2\\test_data", mfcc_test)
    np.save("RoDigits_splitV2\\test_label", labels_test)


if __name__ == '__main__':
    main()
