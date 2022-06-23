# This file is used for evaluating the models' robustness to different black-box and white-box attacks
# To obtain the results shown in the thesis the following inputs should be chosen:
# Black-box attack with white noise added to the MFCCs: a -> b -> s -> m
# Black-box attack with white noise added to the audio: a -> b -> s -> a
# Black-box attack with white noise mixture added to the MFCCs: a -> b -> m -> m
# Black-box attack with white noise mixture added to the audio: a -> b -> m -> a
# Black-box attack with white noise added to the audio waveforms with targeted SNR: a -> b -> snr -> a
# White-box attack FGSM: a -> w -> f
# White-box attack PGD: a -> w -> p
# White-box attack JSMA: a -> w -> j
# White-box attack Carlini l2: a -> w -> l2
# White-box attack Carlini l2: a -> w -> linf

import tensorflow as tf
from art.attacks.evasion import FastGradientMethod, CarliniL2Method, CarliniLInfMethod, ProjectedGradientDescent, \
    SaliencyMapMethod
from art.estimators.classification import TensorFlowV2Classifier
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import librosa


def load_npy_dataset(path):
    """
    Loads the train/dev/test sets that are in a folder in the given path.
    The folder should contain 6 .npy files used for train/dev/test data and train/dev/test labels

    arguments:
    path: path to the folder where the .npy files are contained

    output:
    train_data, train_label, val_data, val_label, test_data, test_label: The loaded data sets and labels
    """
    train_data = np.load(path + "train_data.npy")
    train_label = np.load(path + "train_label.npy")
    val_label = np.load(path + "dev_label.npy")
    val_data = np.load(path + "dev_data.npy")
    test_data = np.load(path + "test_data.npy")
    test_label = np.load(path + "test_label.npy")

    return train_data, train_label, val_data, val_label, test_data, test_label


def standardize_dataset(train_data, val_data, test_data):
    """
    Takes the train/dev/test data and it standardizes them to a normal distribution of mean = 0 and std dev = 1

    arguments:
    train_data: train data set
    val_data: validation data set
    test_data: test data set

    output:
    train_data, val_data, test_data: normalized to N(0,1)
    """
    # Standardizing the data
    all_data = np.concatenate((train_data, val_data, test_data), axis=0)
    scaler1 = StandardScaler()
    all_data = scaler1.fit_transform(all_data)

    train_data = all_data[:train_data.shape[0]]
    val_data = all_data[train_data.shape[0]:train_data.shape[0] + val_data.shape[0]]
    test_data = all_data[train_data.shape[0] + val_data.shape[0]:]

    return train_data, val_data, test_data


# Type one black-box attack
def add_white_noise(array, sigma):
    """
    Adds white gaussian noise on an array with mean = 0 and std dev = sigma

    arguments:
    array: input array
    sigma: the standard deviation of the white noise

    output:
    noisy_array: noisy signal
    """
    noise = np.random.normal(0, sigma, np.array(array).shape[0])
    noisy_array = array + noise
    return noisy_array


def black_box_attack_on_audio(file_path, utterance_length, sigma=0, p=0, alpha=0):
    """
    Adds white gaussian noise on an array using the add_white_noise(array, sigma) function
    or the add_noise(x, p, alpha) function, depending on which input arguments are equal to 0 and then it computes
    the MFCC of the resulted noisy signal.

    arguments:
    file_path: the file path to the audio signal
    utterance_length: the length of the signal in number of windows
    sigma: the standard deviation of the white gaussian noise, if it is 0 then no white noise is added
    p: probability of peaks
    alpha: standard deviation of background noise, if both alpha and p are equal to zero then no mixt noise is added

    output:
    mfcc_features: the MFCC of the noisy audio
    """
    # Get raw .wav data and sampling rate from librosa's load function
    raw_w, sampling_rate = librosa.load(file_path, mono=True)

    if sigma != 0:
        raw_w = add_white_noise(raw_w, sigma)
    elif (p != 0) and (alpha != 0):
        raw_w = add_noise(x=raw_w, p=p, alpha=alpha)

    # Obtain MFCC Features from raw data
    mfcc_features = librosa.feature.mfcc(y=raw_w, sr=sampling_rate)
    if mfcc_features.shape[1] > utterance_length:
        mfcc_features = mfcc_features[:, 0:utterance_length]
    else:
        mfcc_features = np.pad(mfcc_features, ((0, 0), (0, utterance_length - mfcc_features.shape[1])),
                               mode='constant', constant_values=0)

    return mfcc_features


def black_box_attack_on_audio_dataset(filenames, sigma, p, alpha):
    """
    Adds white gaussian noise on an entire data set and returns the MFCC for the whole data set

    arguments:
    filenames: array containing the paths to each audio file in the data set
    sigma: the standard deviation of the white gaussian noise, if it is 0 then no white noise is added
    p: probability of peaks
    alpha: standard deviation of background noise, if both alpha and p are equal to zero then no mixt noise is added

    output:
    mfcc_whole_dataset_flattened: the MFCC for the whole data set
    """
    mfcc_whole_dataset = np.zeros((len(filenames), 20, 44))
    mfcc_whole_dataset_flattened = np.zeros((len(filenames), 20 * 44))
    for index in range(len(filenames)):
        mfcc_whole_dataset[index] = black_box_attack_on_audio(filenames[index], 44, sigma=sigma, p=p, alpha=alpha)
        mfcc_whole_dataset_flattened[index] = mfcc_whole_dataset[index].flatten()
    return mfcc_whole_dataset_flattened


def mixtgauss(N, p, sigma0, sigma1):
    """
    gives a mixtuare of gaussian noise

    arguments:
    N: data length
    p: probability of peaks
    sigma0: standard deviation of backgrond noise
    sigma1: standard deviation of impulse noise

    output:
    x: output noise
    """
    q = np.random.normal(0, 1, N)
    u = np.abs(q) < p
    x = (sigma0 * (1 - u) + sigma1 * u) * np.random.normal(0, 1, N)

    return x


def add_noise(x, p, alpha):
    """
    returns the signal with noise averaged by k

    arguments:
    x: input clean signal
    p: probability of peaks
    alpha: standard deviation of backgrond noise

    outputs:
    x_noisy: noisy signal
    """
    N = x.shape[0]
    sigma0 = alpha
    sigma1 = 10 * alpha
    noise = mixtgauss(N, p, sigma0, sigma1)
    x_noisy = x + noise

    return x_noisy


def add_white_noise_on_dataset(dataset, sigma):
    """
    Adds white gaussian noise on an entire data set and returns the MFCC for the whole data set
    This is used to add white noise directly on MFCC

    arguments:
    dataset: array containing the paths to each audio file in the data set
    sigma: the standard deviation of the white gaussian noise, if it is 0 then no white noise is added

    output:
    noisy_dataset: MFCC data set with added noise
    """
    noisy_dataset = np.array(dataset)
    for index in range(noisy_dataset.shape[0]):
        noisy_dataset[index] = add_white_noise(noisy_dataset[index], sigma)
    return noisy_dataset


def add_noise_mixture_on_dataset(dataset, p, alpha):
    """
    Adds white gaussian noise mixture on an entire data set and returns the MFCC for the whole data set
    This is used to add white noise directly on MFCC

    arguments:
    dataset: array containing the paths to each audio file in the data set
    sigma: the standard deviation of the white gaussian noise, if it is 0 then no white noise is added

    output:
    noisy_dataset: MFCC data set with added noise mixture
    """
    noisy_dataset = np.array(dataset)
    for index in range(noisy_dataset.shape[0]):
        noisy_dataset[index] = add_noise(x=noisy_dataset[index], p=p, alpha=alpha)
    return noisy_dataset


def add_white_noise_with_snr(audio, target_snr_db):
    """
    Adds white gaussian noise on an audio file, it adds the noise with respect to the target_snr_db

    arguments:
    audio: audio file as an array
    target_snr_db: the target signal to noise ratio in dB

    output:
    noisy_signal: noisy signal
    """
    sample = np.asanyarray(audio)
    signal_avg_watts = np.mean(sample ** 2)
    signal_avg_db = 10 * np.log10(signal_avg_watts)
    noise_avg_db = signal_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    mean_noise = 0
    # k = np.sqrt(1 / (10 ** (target_snr_db / 10))) // if norming to the power of the signal is wanted
    k = 1
    noise_volts = k * np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(sample))
    noise_volts_power = np.mean(noise_volts ** 2)
    noise_db = 10 * np.log10(noise_volts_power)
    noisy_signal = sample + noise_volts
    return noisy_signal


def black_box_attack_on_audio_snr(file_path, utterance_length, target_snr_db):
    """
    Adds white gaussian noise on an audio file, it adds the noise with respect to the target_snr_db using the
    add_white_noise_with_snr(audio, target_snr_db) function and the computes the MFCC features for that file

    arguments:
    file_path: the file path to the audio signal
    utterance_length: the length of the signal in number of windows
    target_snr_db: the target signal to noise ratio in dB

    output:
    mfcc_features: the MFCC of the noisy audio
    """
    # Get raw .wav data and sampling rate from librosa's load function
    raw_w, sampling_rate = librosa.load(file_path, mono=True)

    raw_w = add_white_noise_with_snr(raw_w, target_snr_db)

    # Obtain MFCC Features from raw data
    mfcc_features = librosa.feature.mfcc(y=raw_w, sr=sampling_rate)
    if mfcc_features.shape[1] > utterance_length:
        mfcc_features = mfcc_features[:, 0:utterance_length]
    else:
        mfcc_features = np.pad(mfcc_features, ((0, 0), (0, utterance_length - mfcc_features.shape[1])),
                               mode='constant', constant_values=0)

    return mfcc_features


def black_box_attack_on_audio_dataset_snr(filenames, target_snr_db):
    """
    Adds white gaussian noise on an audio data set, it adds the noise with respect to the target_snr_db using the
    add_white_noise_with_snr(audio, target_snr_db) function and the computes the MFCC features for each file

    arguments:
    filenames: array containing the paths to each audio file in the data set
    target_snr_db: the target signal to noise ratio in dB

    output:
    mfcc_whole_dataset_flattened: the MFCC for the whole data set
    """
    mfcc_whole_dataset = np.zeros((len(filenames), 20, 44))
    mfcc_whole_dataset_flattened = np.zeros((len(filenames), 20 * 44))
    for index in range(len(filenames)):
        mfcc_whole_dataset[index] = black_box_attack_on_audio_snr(filenames[index], 44, target_snr_db)
        mfcc_whole_dataset_flattened[index] = mfcc_whole_dataset[index].flatten()
    return mfcc_whole_dataset_flattened


if __name__ == '__main__':

    # Loading the datasets
    # For the attacks we only need the test dataset, but standardizing has to be done in the same method
    path = 'processed_google_dataset/'

    test_filenames = np.load("test_dataset_to_add_noise\\test_filenames.npy")
    test_labels = np.load("test_dataset_to_add_noise\\test_label.npy")
    test_labels2 = []
    for elem in test_labels:
        test_labels2.append(str(elem))
    test_labels2 = np.array(test_labels2)
    test_labels = to_categorical(test_labels, 10)

    train_data, train_label, val_data, val_label, test_data, test_label = load_npy_dataset(path)
    test_label1 = to_categorical(test_label, 10)
    print(np.max(test_data))

    model_constrained = load_model("bin/models_constrained/model_constrained_Rho01_dropout01.h5")

    model_unconstrained = load_model("bin/models/baseline.h5")

    SNRs = [60, 30, 20, 15, 10, 5, 0]  # the values are in dB
    sigmas = np.linspace(0, 100, 20)
    alphas = np.linspace(0, 0.015, 15)
    accuracy_constrained = []
    accuracy_unconstrained = []

    attack_after_standardization = input("Should the data be standardized before or after the attack? [B]/[A] ").lower()
    if attack_after_standardization == 'b':
        train_data, val_data, test_data = standardize_dataset(train_data, val_data, test_data)

    type_of_attack = input("Black-box or white-box attack? [B]/[W] ").lower()

    if type_of_attack == 'b':
        type_of_black_box_attack = input("Type of black-box attack [S]imple/[M]ixture/[SNR]: ").lower()
        noise_over_audio_or_mfcc = input("Add white noise over [A]udio or [M]FCC: ").lower()
        if noise_over_audio_or_mfcc == 'a':
            sigmas = [0, 0.002, 0.004, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
            if type_of_black_box_attack == 's':
                for sigma in sigmas:
                    # Here the data is raw audio data at the input, so there is no need for standardizing it
                    test_data2 = black_box_attack_on_audio_dataset(test_filenames, sigma, p=0, alpha=0)

                    # Now we standardize the data
                    _, _, test_data2 = standardize_dataset(train_data, val_data, test_data2)

                    predictions_constrained = model_constrained.predict(test_data2)
                    predictions_unconstrained = model_unconstrained.predict(test_data2)

                    accuracy_constrained1 = np.sum(
                        np.argmax(predictions_constrained, axis=1) == np.argmax(test_labels, axis=1)) / len(test_labels)
                    accuracy_constrained = np.append(accuracy_constrained, accuracy_constrained1)
                    print("Accuracy on black-box attack test examples: {}%".format(accuracy_constrained1 * 100))

                    accuracy_unconstrained1 = np.sum(
                        np.argmax(predictions_unconstrained, axis=1) == np.argmax(test_labels, axis=1)) / len(
                        test_labels)
                    accuracy_unconstrained = np.append(accuracy_unconstrained, accuracy_unconstrained1)
                    print("Accuracy on black-box attack test examples unconstrained: {}%".format(
                        accuracy_unconstrained1 * 100))

                fig, ax = plt.subplots()
                ax.plot(sigmas, accuracy_constrained, color='r', label='Constrained Model')
                ax.plot(sigmas, accuracy_unconstrained, color='b', label='Unconstrained model')
                ax.legend()
                ax.set_title('Accuracy vs Noise Sigma')
                ax.set_xlabel('Sigma')
                ax.set_ylabel('Accuracy')
                plt.show()

            elif type_of_black_box_attack == 'm':
                p = 0.01
                for alpha in alphas:
                    # Here the data is raw audio data at the input, so there is no need for standardizing it
                    test_data2 = black_box_attack_on_audio_dataset(test_filenames, sigma = 0, p=p, alpha=alpha)

                    # Now we standardize the data
                    _, _, test_data2 = standardize_dataset(train_data, val_data, test_data2)

                    predictions_constrained = model_constrained.predict(test_data2)
                    predictions_unconstrained = model_unconstrained.predict(test_data2)

                    accuracy_constrained1 = np.sum(
                        np.argmax(predictions_constrained, axis=1) == np.argmax(test_label1, axis=1)) / len(test_label1)
                    accuracy_constrained = np.append(accuracy_constrained, accuracy_constrained1)
                    print("Accuracy on black-box attack test examples: {}%".format(accuracy_constrained1 * 100))

                    accuracy_unconstrained1 = np.sum(
                        np.argmax(predictions_unconstrained, axis=1) == np.argmax(test_label1, axis=1)) / len(
                        test_label1)
                    accuracy_unconstrained = np.append(accuracy_unconstrained, accuracy_unconstrained1)
                    print("Accuracy on black-box attack test examples unconstrained: {}%".format(
                        accuracy_unconstrained1 * 100))

                fig, ax = plt.subplots()
                ax.plot(alphas, accuracy_constrained, color='r', label='Constrained Model')
                ax.plot(alphas, accuracy_unconstrained, color='b', label='Unconstrained model')
                ax.legend()
                ax.set_title('Accuracy vs Alpha')
                ax.set_xlabel('Alpha')
                ax.set_ylabel('Accuracy')
                plt.show()

            elif type_of_black_box_attack == 'snr':
                for snr in SNRs:
                    # Here the data is raw audio data at the input, so there is no need for standardizing it
                    test_data2 = black_box_attack_on_audio_dataset_snr(test_filenames, snr)

                    # Now we standardize the data
                    _, _, test_data2 = standardize_dataset(train_data, val_data, test_data2)

                    predictions_constrained = model_constrained.predict(test_data2)
                    predictions_unconstrained = model_unconstrained.predict(test_data2)

                    accuracy_constrained1 = np.sum(
                        np.argmax(predictions_constrained, axis=1) == np.argmax(test_label1, axis=1)) / len(test_label1)
                    accuracy_constrained = np.append(accuracy_constrained, accuracy_constrained1)
                    print("Accuracy on black-box attack test examples: {}%".format(accuracy_constrained1 * 100))

                    accuracy_unconstrained1 = np.sum(
                        np.argmax(predictions_unconstrained, axis=1) == np.argmax(test_label1, axis=1)) / len(
                        test_label1)
                    accuracy_unconstrained = np.append(accuracy_unconstrained, accuracy_unconstrained1)
                    print("Accuracy on black-box attack test examples unconstrained: {}%".format(
                        accuracy_unconstrained1 * 100))

                fig, ax = plt.subplots()
                ax.plot(SNRs, accuracy_constrained, color='r', label='Constrained Model')
                ax.plot(SNRs, accuracy_unconstrained, color='b', label='Unconstrained model')
                ax.legend()
                ax.set_title('Accuracy vs SNR')
                ax.set_xlabel('SNR [dB]')
                ax.set_ylabel('Accuracy')
                plt.show()

        elif noise_over_audio_or_mfcc == 'm':
            if type_of_black_box_attack == 's':
                for sigma in sigmas:
                    test_data2 = add_white_noise_on_dataset(test_data, sigma)
                    if attack_after_standardization == 'a':
                        _, _, test_data2 = standardize_dataset(train_data, val_data, test_data2)

                    predictions_constrained = model_constrained.predict(test_data2)
                    predictions_unconstrained = model_unconstrained.predict(test_data2)

                    accuracy_constrained1 = np.sum(
                        np.argmax(predictions_constrained, axis=1) == np.argmax(test_label1, axis=1)) / len(test_label1)
                    accuracy_constrained = np.append(accuracy_constrained, accuracy_constrained1)
                    print("Accuracy on black-box attack test examples: {}%".format(accuracy_constrained1 * 100))

                    accuracy_unconstrained1 = np.sum(
                        np.argmax(predictions_unconstrained, axis=1) == np.argmax(test_label1, axis=1)) / len(test_label1)
                    accuracy_unconstrained = np.append(accuracy_unconstrained, accuracy_unconstrained1)
                    print("Accuracy on black-box attack test examples unconstrained: {}%".format(accuracy_unconstrained1 * 100))

                fig, ax = plt.subplots()
                ax.plot(sigmas, accuracy_constrained, color='r', label='Constrained Model')
                ax.plot(sigmas, accuracy_unconstrained, color='b', label='Unconstrained model')
                ax.legend()
                ax.set_title('Accuracy vs Noise Sigma')
                ax.set_xlabel('Sigma')
                ax.set_ylabel('Accuracy')
                plt.show()

            elif type_of_black_box_attack == 'm':
                p = 0.01
                alphas = np.linspace(0, 100, 30)
                for alpha in alphas:
                    test_data2 = add_noise_mixture_on_dataset(dataset=test_data, p=p, alpha=alpha)
                    if attack_after_standardization == 'a':
                        _, _, test_data2 = standardize_dataset(train_data, val_data, test_data2)

                    predictions_constrained = model_constrained.predict(test_data2)
                    predictions_unconstrained = model_unconstrained.predict(test_data2)

                    accuracy_constrained1 = np.sum(
                        np.argmax(predictions_constrained, axis=1) == np.argmax(test_label1, axis=1)) / len(test_label1)
                    accuracy_constrained = np.append(accuracy_constrained, accuracy_constrained1)
                    print("Accuracy on black-box attack test examples: {}%".format(accuracy_constrained1 * 100))

                    accuracy_unconstrained1 = np.sum(
                        np.argmax(predictions_unconstrained, axis=1) == np.argmax(test_label1, axis=1)) / len(test_label1)
                    accuracy_unconstrained = np.append(accuracy_unconstrained, accuracy_unconstrained1)
                    print("Accuracy on black-box attack test examples unconstrained: {}%".format(
                        accuracy_unconstrained1 * 100))

                fig, ax = plt.subplots()
                ax.plot(alphas, accuracy_constrained, color='r', label='Constrained Model')
                ax.plot(alphas, accuracy_unconstrained, color='b', label='Unconstrained model')
                ax.legend()
                ax.set_title('Accuracy vs Alpha')
                ax.set_xlabel('Alpha')
                ax.set_ylabel('Accuracy')
                plt.show()

    elif type_of_attack == 'w':
        type_of_white_box_attack = input("Type of white box attack: [F]GSM/Carlini[L2]/Carlini[Linf]/[P]GD:/[J]SMA ").lower()

        if type_of_white_box_attack == 'f':
            eps = np.linspace(0.01, 0.3, 10)
            if attack_after_standardization == 'a':
                eps = np.linspace(1, 30, 50)
            model_constrained = TensorFlowV2Classifier(model=model_constrained, nb_classes=10, input_shape=(880,)
                                                       , loss_object=CategoricalCrossentropy())

            model_unconstrained = TensorFlowV2Classifier(model=model_unconstrained, nb_classes=10, input_shape=(880,)
                                                         , loss_object=CategoricalCrossentropy())
            for item in eps:
                attack_constrained = FastGradientMethod(estimator=model_constrained, eps=item)
                attack_unconstrained = FastGradientMethod(estimator=model_unconstrained, eps=item)

                test_adv_constrained = attack_constrained.generate(x=np.array(test_data))
                test_adv_unconstrained = attack_unconstrained.generate(x=test_data)

                if attack_after_standardization == 'a':
                    _, _, test_adv_constrained = standardize_dataset(train_data, val_data,
                                                                     test_adv_constrained)
                    _, _, test_adv_unconstrained = standardize_dataset(train_data, val_data,
                                                                       test_adv_unconstrained)

                predictions_constrained = model_constrained.predict(test_adv_constrained)
                predictions_unconstrained = model_unconstrained.predict(test_adv_unconstrained)

                accuracy_constrained1 = np.sum(np.argmax(predictions_constrained, axis=1) == np.argmax(test_label1, axis=1)) / len(test_label1)
                accuracy_constrained = np.append(accuracy_constrained, accuracy_constrained1)
                print("Accuracy on adversarial test examples: {}%".format(accuracy_constrained1 * 100))

                accuracy_unconstrained1 = np.sum(np.argmax(predictions_unconstrained, axis=1) == np.argmax(test_label1, axis=1)) / len(test_label1)
                accuracy_unconstrained = np.append(accuracy_unconstrained, accuracy_unconstrained1)
                print("Accuracy on adversarial test examples unconstrained: {}%".format(accuracy_unconstrained1 * 100))

            fig, ax = plt.subplots()
            ax.plot(eps, accuracy_constrained, color='r', label='Constrained Model')
            ax.plot(eps, accuracy_unconstrained, color='b', label='Unconstrained model')
            ax.legend()
            ax.set_title('Accuracy vs Attack epsilon FGSM attack')
            ax.set_xlabel('Epsilon')
            ax.set_ylabel('Accuracy')
            plt.show()

        elif type_of_white_box_attack == 'j':
            eps = [10]
            model_constrained = TensorFlowV2Classifier(model=model_constrained, nb_classes=10, input_shape=(880,)
                                                       , loss_object=CategoricalCrossentropy())

            model_unconstrained = TensorFlowV2Classifier(model=model_unconstrained, nb_classes=10, input_shape=(880,)
                                                         , loss_object=CategoricalCrossentropy())
            for item in eps:
                attack_unconstrained = SaliencyMapMethod(classifier=model_unconstrained, theta=item, gamma=0.1)
                attack_constrained = SaliencyMapMethod(classifier=model_constrained, theta=item, gamma=0.1)

                test_adv_unconstrained = attack_unconstrained.generate(x=test_data[:100])
                test_adv_constrained = attack_constrained.generate(x=np.array(test_data[:100]))

                if attack_after_standardization == 'a':
                    train_data, val_data, test_adv_constrained = standardize_dataset(train_data, val_data,
                                                                                     test_adv_constrained)
                    train_data, val_data, test_adv_unconstrained = standardize_dataset(train_data, val_data,
                                                                                       test_adv_unconstrained)

                predictions_constrained = model_constrained.predict(test_adv_constrained)
                predictions_unconstrained = model_unconstrained.predict(test_adv_unconstrained)

                accuracy_constrained1 = np.sum(np.argmax(predictions_constrained, axis=1) ==
                                               np.argmax(test_label1[:100], axis=1)) / len(test_label1[:100])
                accuracy_constrained = np.append(accuracy_constrained, accuracy_constrained1)
                print("Accuracy on adversarial test examples: {}%".format(accuracy_constrained1 * 100))

                accuracy_unconstrained1 = np.sum(np.argmax(predictions_unconstrained, axis=1) ==
                                                 np.argmax(test_label1[:100], axis=1)) / len(test_label1[:100])
                accuracy_unconstrained = np.append(accuracy_unconstrained, accuracy_unconstrained1)
                print("Accuracy on adversarial test examples unconstrained: {}%".format(accuracy_unconstrained1 * 100))

        elif type_of_white_box_attack == 'linf':
            eps = [10]
            model_constrained = TensorFlowV2Classifier(model=model_constrained, nb_classes=10, input_shape=(880,)
                                                       , loss_object=CategoricalCrossentropy())

            model_unconstrained = TensorFlowV2Classifier(model=model_unconstrained, nb_classes=10, input_shape=(880,)
                                                         , loss_object=CategoricalCrossentropy())
            for item in eps:
                attack_constrained = CarliniLInfMethod(classifier=model_constrained, confidence=item)
                attack_unconstrained = CarliniLInfMethod(classifier=model_unconstrained, confidence=item)

                test_adv_constrained = attack_constrained.generate(x=np.array(test_data))
                test_adv_unconstrained = attack_unconstrained.generate(x=test_data)

                if attack_after_standardization == 'a':
                    train_data, val_data, test_adv_constrained = standardize_dataset(train_data, val_data,
                                                                                     test_adv_constrained)
                    train_data, val_data, test_adv_unconstrained = standardize_dataset(train_data, val_data,
                                                                                       test_adv_unconstrained)

                predictions_constrained = model_constrained.predict(test_adv_constrained)
                predictions_unconstrained = model_unconstrained.predict(test_adv_unconstrained)

                accuracy_constrained1 = np.sum(
                    np.argmax(predictions_constrained, axis=1) ==
                    np.argmax(test_label1, axis=1)) / len(test_label1)
                accuracy_constrained = np.append(accuracy_constrained, accuracy_constrained1)
                print("Accuracy on adversarial test examples: {}%".format(accuracy_constrained1 * 100))

                accuracy_unconstrained1 = np.sum(
                    np.argmax(predictions_unconstrained, axis=1) ==
                    np.argmax(test_label1, axis=1)) / len(test_label1)
                accuracy_unconstrained = np.append(accuracy_unconstrained, accuracy_unconstrained1)
                print("Accuracy on adversarial test examples unconstrained: {}%".format(accuracy_unconstrained1 * 100))

        elif type_of_white_box_attack == 'l2':
            eps = np.linspace(1, 300, 3)
            model_constrained = TensorFlowV2Classifier(model=model_constrained, nb_classes=10, input_shape=(880,)
                                                       , loss_object=CategoricalCrossentropy())

            model_unconstrained = TensorFlowV2Classifier(model=model_unconstrained, nb_classes=10, input_shape=(880,)
                                                         , loss_object=CategoricalCrossentropy())
            for item in eps:

                attack_constrained = CarliniL2Method(classifier=model_constrained, confidence=item)
                attack_unconstrained = CarliniL2Method(classifier=model_unconstrained, confidence=item)

                targets = np.copy(test_label1)
                np.random.shuffle(targets)

                test_adv_constrained = attack_constrained.generate(x=test_data)
                test_adv_unconstrained = attack_unconstrained.generate(x=test_data)

                if attack_after_standardization == 'a':
                    train_data, val_data, test_adv_constrained = standardize_dataset(train_data, val_data,
                                                                                     test_adv_constrained)
                    train_data,val_data, test_adv_unconstrained = standardize_dataset(train_data, val_data,
                                                                                      test_adv_unconstrained)

                predictions_constrained = model_constrained.predict(test_adv_constrained)
                predictions_unconstrained = model_unconstrained.predict(test_adv_unconstrained)

                accuracy_constrained1 = np.sum(
                    np.argmax(predictions_constrained, axis=1) ==
                    np.argmax(test_label1, axis=1)) / len(test_label1)
                accuracy_constrained = np.append(accuracy_constrained, accuracy_constrained1)
                print(f"Carlini L2 with confidence={item} accuracy on adversarial test examples: "
                      f"{accuracy_constrained1 * 100}%")

                accuracy_unconstrained1 = np.sum(
                    np.argmax(predictions_unconstrained, axis=1) ==
                    np.argmax(test_label1, axis=1)) / len(test_label1)
                accuracy_unconstrained = np.append(accuracy_unconstrained, accuracy_unconstrained1)
                print(f"Carlini L2 with confidence={item} accuracy on adversarial test examples unconstrained: "
                      f"{accuracy_unconstrained1 * 100}%")

        elif type_of_white_box_attack == 'p':
            eps = np.linspace(1, 30, 50)

            model_constrained = TensorFlowV2Classifier(model=model_constrained, nb_classes=10, input_shape=(880,)
                                                       , loss_object=CategoricalCrossentropy())

            model_unconstrained = TensorFlowV2Classifier(model=model_unconstrained, nb_classes=10, input_shape=(880,)
                                                         , loss_object=CategoricalCrossentropy())

            for item in eps:
                attack_constrained = ProjectedGradientDescent(estimator=model_constrained, eps=item)
                attack_unconstrained = ProjectedGradientDescent(estimator=model_unconstrained, eps=item)

                test_adv_constrained = attack_constrained.generate(x=test_data)
                test_adv_unconstrained = attack_unconstrained.generate(x=test_data)

                if attack_after_standardization == 'a':
                    _, _, test_adv_constrained = standardize_dataset(train_data, val_data,
                                                                     test_adv_constrained)
                    _, _, test_adv_unconstrained = standardize_dataset(train_data, val_data,
                                                                       test_adv_unconstrained)

                predictions_constrained = model_constrained.predict(test_adv_constrained)
                predictions_unconstrained = model_unconstrained.predict(test_adv_unconstrained)

                accuracy_constrained1 = np.sum(
                    np.argmax(predictions_constrained, axis=1) ==
                    np.argmax(test_label1, axis=1)) / len(test_label1)
                accuracy_constrained = np.append(accuracy_constrained, accuracy_constrained1)
                print(f"PGD attack with eps={item} accuracy on adversarial test examples: "
                      f"{accuracy_constrained1 * 100}%")

                accuracy_unconstrained1 = np.sum(
                    np.argmax(predictions_unconstrained, axis=1) ==
                    np.argmax(test_label1, axis=1)) / len(test_label1)
                accuracy_unconstrained = np.append(accuracy_unconstrained, accuracy_unconstrained1)
                print(f"PGD attack with eps={item} Accuracy on adversarial test examples unconstrained: "
                      f"{accuracy_unconstrained1 * 100}%")

            fig, ax = plt.subplots()
            ax.plot(eps, accuracy_constrained, color='r', label='Constrained Model')
            ax.plot(eps, accuracy_unconstrained, color='b', label='Unconstrained model')
            ax.legend()
            ax.set_title('Accuracy Projected Gradient Descent attack')
            ax.set_xlabel('Epsilon')
            ax.set_ylabel('Accuracy')
            plt.show()
