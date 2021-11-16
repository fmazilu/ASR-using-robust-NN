import tensorflow as tf
from art.attacks.evasion import FastGradientMethod, CarliniL2Method
from art.estimators.classification import TensorFlowClassifier, KerasClassifier, TensorFlowV2Classifier
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import librosa
from Constraints import customConstraint
# from extract_features_construct_dataset import get_lipschitz_constrained, get_upper_lipschitz, get_norms
# from mixtgauss import add_noise

def load_npy_dataset(path):
    train_data = np.load(path + "train_data.npy")
    train_label = np.load(path + "train_label.npy")
    # train_label = to_categorical(train_label, 10)
    val_label = np.load(path + "dev_label.npy")
    # val_label = to_categorical(val_label, 10)
    val_data = np.load(path + "dev_data.npy")
    test_data = np.load(path + "test_data.npy")
    test_label = np.load(path + "test_label.npy")
    # test_label = to_categorical(test_label1, 10)

    return train_data, train_label, val_data, val_label, test_data, test_label

def standardize_dataset(train_data, val_data, test_data):
    # Standardizing the data
    all_data = np.concatenate((train_data, val_data, test_data), axis=0)
    scaler1 = StandardScaler()
    all_data = scaler1.fit_transform(all_data)

    train_data = all_data[:train_data.shape[0]]
    val_data = all_data[train_data.shape[0]:train_data.shape[0] + val_data.shape[0]]
    test_data = all_data[train_data.shape[0] + val_data.shape[0]:]

    return train_data, val_data, test_data

def add_white_noise(array, sigma):
    noise = np.random.normal(0, sigma, np.array(array).shape[0])
    noisy_array = array + noise
    return noisy_array


def mixtgauss(N, p, sigma0, sigma1):
    """
    gives a mixtuare of gaussian noise
    arguments:
    N: data length
    p: probability of peaks
    sigma0: standard deviation of backgrond noise
    sigma1: standard deviation of impulse noise

    output: x: output noise
    """
    q = np.random.randn(N, 1)
    # print(q)
    # print(q.shape)
    u = q < p
    # print(u)
    # print(sigma1 * u)
    # print(1-u)
    x = (sigma0 * (1 - u) + sigma1 * u) * np.random.randn(N, 1)

    return x


def add_noise(x, p, alpha):
    '''
    returns the signal with noise averaged by k

    arguments:
    x: input clean signal
    p: probability of peaks
    alpha: standard deviation of backgrond noise

    outputs:
    x_noisy: noisy signal

    '''
    N = x.shape[0]
    sigma0 = alpha
    sigma1 = 10 * alpha

    noise = mixtgauss(N, p, sigma0, sigma1)

    x_noisy = x + noise

    return x_noisy


def add_white_noise_on_dataset(dataset, sigma):
    noisy_dataset = np.array(dataset)
    for index in range(noisy_dataset.shape[0]):
        noisy_dataset[index] = add_white_noise(noisy_dataset[index], sigma)
    return noisy_dataset


def add_noise_mixture_on_dataset(dataset, p, alpha):
    noisy_dataset = np.array(dataset)
    # print(noisy_dataset.shape) (2366, 880)
    for index in range(noisy_dataset.shape[0]):
        # print(noisy_dataset[index].shape)  #(880,)
        noisy_dataset[index] = add_noise(x=np.expand_dims(noisy_dataset[index], axis=0), p=p, alpha=alpha)
    return noisy_dataset


def black_box_attack_on_audio(file_path, utterance_length, sigma = 0, p = 0, alpha = 0):
    # Get raw .wav data and sampling rate from librosa's load function
    raw_w, sampling_rate = librosa.load(file_path, mono=True)

    if sigma != 0:
        raw_w = add_white_noise(raw_w, sigma)
    elif (p != 0) and (alpha != 0):
        raw_w = add_noise(x=np.expand_dims(raw_w, axis=0), p=p, alpha=alpha)
        raw_w = np.transpose(raw_w)
        raw_w = raw_w.flatten()
    # else:
        # print('There were no valid arguments for adding noise.')

    # Obtain MFCC Features from raw data
    mfcc_features = librosa.feature.mfcc(raw_w, sampling_rate)
    if mfcc_features.shape[1] > utterance_length:
        mfcc_features = mfcc_features[:, 0:utterance_length]
    else:
        mfcc_features = np.pad(mfcc_features, ((0, 0), (0, utterance_length - mfcc_features.shape[1])),
                               mode='constant', constant_values=0)

    return mfcc_features


def black_box_attack_on_audio_dataset(filenames, sigma, p, alpha):
    mfcc_whole_dataset = np.zeros((len(filenames), 20, 44))
    mfcc_whole_dataset_flattened = np.zeros((len(filenames), 20 * 44))
    for index in range(len(filenames)):
        mfcc_whole_dataset[index] = black_box_attack_on_audio(filenames[index], 44, sigma=sigma, p=p, alpha=alpha)
        mfcc_whole_dataset_flattened[index] = mfcc_whole_dataset[index].flatten()
    return mfcc_whole_dataset_flattened


if __name__ == '__main__':
    # Loading the datasets
    ## For the attacks we only need the test dataset, but standardizing has to be done in the same method
    path = 'processed_google_dataset/'

    test_filenames = np.load("test_dataset_to_add_noise\\test_filenames.npy")
    test_labels = np.load("test_dataset_to_add_noise\\test_label.npy")
    test_labels = to_categorical(test_labels, 10)

    train_data, train_label, val_data, val_label, test_data, test_label = load_npy_dataset(path)
    test_label1 = to_categorical(test_label, 10)

    model_constrained = load_model("bin/models_constrained/model_constrained_Rho01_dropout01.h5")
                                   # custom_objects={'customConstraint': customConstraint})
    model_unconstrained = load_model("bin/models/baseline.h5")

    # TODO: Move these commented lines to another script
    # lip_cst = get_lipschitz_constrained(model_constrained)
    # print(f"The Lipschitz Constant for the constrained model: {lip_cst}")
    # norms = get_norms(model_constrained)
    # upper_lip = get_upper_lipschitz(norms)
    # print(f"The upper-bound Lipschitz Constant for the constrained model: {upper_lip}")
    # norms = get_norms(model_unconstrained)
    # upper_lip = get_upper_lipschitz(norms)
    # print(f"The upper-bound Lipschitz Constant for the unconstrained model: {upper_lip}")

    sigmas = np.linspace(0, 10, 10)
    alphas = np.linspace(0.01, 2, 20)
    accuracy_constrained = []
    accuracy_unconstrained = []

    attack_after_standardization = input("Should the data be standardized before or after the attack? [B]/[A] ").lower()
    if attack_after_standardization == 'b':
        train_data, val_data, test_data = standardize_dataset(train_data, val_data, test_data)

    type_of_attack = input("Black-box or white-box attack? [B]/[W] ").lower()

    if type_of_attack == 'b':
        type_of_black_box_attack = input("Type of black-box attack [S]imple/[M]ixture: ").lower()
        noise_over_audio_or_mfcc = input("Add white noise over [A]udio or [M]FCC: ").lower()
        if noise_over_audio_or_mfcc == 'a':
            sigmas = np.linspace(0, 0.1, 10)
            if type_of_black_box_attack == 's':
                for sigma in sigmas:
                    # Here the data is raw audio data at the input, so there is no need for standardizing it
                    test_data2 = black_box_attack_on_audio_dataset(test_filenames, sigma, p=0, alpha=0)

                    # Now we standardize the data
                    train_data, val_data, test_data2 = standardize_dataset(train_data, val_data, test_data2)

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

            elif type_of_black_box_attack == 'm':  ## aici de introdus atacul pe baza de mixtura
                p = 0.1
                for alpha in alphas:
                    # Here the data is raw audio data at the input, so there is no need for standardizing it
                    test_data2 = black_box_attack_on_audio_dataset(test_filenames, sigma = 0, p=p, alpha=alpha)

                    # Now we standardize the data
                    train_data, val_data, test_data2 = standardize_dataset(train_data, val_data, test_data2)

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

        elif noise_over_audio_or_mfcc == 'm':
            if type_of_black_box_attack == 's':
                for sigma in sigmas:
                    test_data2 = add_white_noise_on_dataset(test_data, sigma)
                    if attack_after_standardization == 'a':
                        train_data, val_data, test_data2 = standardize_dataset(train_data, val_data, test_data2)

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

            elif type_of_black_box_attack == 'm': ## aici de introdus atacul pe baza de mixtura
                p = 0.1
                for alpha in alphas:
                    test_data2 = add_noise_mixture_on_dataset(dataset=test_data, p=p, alpha=alpha)
                    if attack_after_standardization == 'a':
                        train_data, val_data, test_data2 = standardize_dataset(train_data, val_data, test_data2)

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
        ## de continuat cu atacuri de tip white-box
        eps = np.linspace(0.01, 0.3, 10)
        if attack_after_standardization == 'a':
            eps = np.linspace(1, 10, 10)
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
                train_data, val_data, test_adv_constrained = standardize_dataset(train_data, val_data,
                                                                                 test_adv_constrained)
                train_data, val_data, test_adv_unconstrained = standardize_dataset(train_data, val_data,
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
        ax.set_title('Accuracy vs Attack epsilon')
        ax.set_xlabel('Epsilon')
        ax.set_ylabel('Accuracy')
        plt.show()

        ### Add carlini method to white box attacks from hidden commands paper
        ### TODO: Add gaussian noise over audio and (as putea sa iau inregistrari random, adica nu cele din setul de test? NU)
        # mfcc's and try these attacks, try adding noise before standardization ------> DONE
        ### p=0.1 ; 0.3, 0.4 -> grafice ca in tranpami                          ------> DONE
        ### alpha = 0.1 - 2 ; pas 0.1                                           ------> DONE
        ### TODO: antrenat cu zgomot (adversarial training) un model neconstrans
        ### TODO: sa vad daca mai pot adauga exemple in setul de date in timp ce invata

        #### De refacut seturile de antrenament, validare si testare, salvand fisierele audio de testare -> DONE
        #### TODO: ca sa pot pune zgomot alb peste ele
        #### TODO: intrab-o pe profa daca ar trebui sa normalizez audio inainte de a face MFCC?
        #### pentru ca range-ul de valori depinde de la audio la audio