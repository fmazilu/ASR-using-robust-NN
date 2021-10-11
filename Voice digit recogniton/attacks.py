import tensorflow as tf
from art.attacks.evasion import FastGradientMethod, CarliniL2Method
from art.estimators.classification import TensorFlowClassifier, KerasClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from Constraints import customConstraint
from tensorflow.python.framework.ops import disable_eager_execution


def add_white_noise(array, sigma):
    noise = np.random.normal(0, sigma, np.array(array).shape[0])
    noisy_array = array + noise
    return noisy_array


def add_white_noise_on_dataset(dataset, sigma):
    noisy_dataset = np.array(dataset)
    for index in range(dataset.shape[0]):
        noisy_dataset[index] = add_white_noise(noisy_dataset[index], sigma)
    return noisy_dataset


if __name__ == '__main__':
    # Loading the datasets
    ## For the attacks we only need the test dataset, but standardizing has to be done in the same method
    path = 'processed_google_dataset/'
    train_data = np.load(path + "train_data.npy")
    train_label = np.load(path + "train_label.npy")
    train_label = to_categorical(train_label, 10)
    val_label = np.load(path + "dev_label.npy")
    val_label = to_categorical(val_label, 10)
    val_data = np.load(path + "dev_data.npy")
    test_data = np.load(path + "test_data.npy")
    test_label1 = np.load(path + "test_label.npy")
    test_label = to_categorical(test_label1, 10)

    # Standardizing the data
    all_data = np.concatenate((train_data, val_data, test_data), axis=0)
    scaler1 = StandardScaler()
    all_data = scaler1.fit_transform(all_data)

    train_data = all_data[:train_data.shape[0]]
    val_data = all_data[train_data.shape[0]:train_data.shape[0] + val_data.shape[0]]
    test_data = all_data[train_data.shape[0] + val_data.shape[0]:]

    model_constrained = load_model("bin/models_constrained/model_3layerFISTA_rho5_droupout.h5")
    model_unconstrained = load_model("bin/models/model_dropout0.5.h5")
    sigmas = np.linspace(0, 10, 10)
    accuracy_constrained = []
    accuracy_unconstrained = []

    type_of_attack = input("Black-box or white-box attack? [B]/[W] ").lower()

    if type_of_attack == 'b':
        for sigma in sigmas:
            test_data2 = add_white_noise_on_dataset(test_data, sigma)
            y = np.argmax(model_unconstrained.predict(test_data2), axis=1)
            results = model_unconstrained.evaluate(test_data2, test_label)
            # print(f'Test loss: {results[0]} / Test accuracy: {results[1]}')
            accuracy_unconstrained = np.append(accuracy_unconstrained, results[1])

        print()

        for sigma in sigmas:
            test_data1 = add_white_noise_on_dataset(test_data, sigma)
            y = np.argmax(model_constrained.predict(test_data1), axis=1)
            results = model_constrained.evaluate(test_data1, test_label)
            # print(f'Test loss: {results[0]} / Test accuracy: {results[1]}')
            accuracy_constrained = np.append(accuracy_constrained, results[1])

        fig, ax = plt.subplots()
        ax.plot(sigmas, accuracy_constrained, color='r', label='Constrained Model')
        ax.plot(sigmas, accuracy_unconstrained, color='b', label='Unconstrained Model')
        ax.legend()
        ax.set_title('Accuracy vs noise sigma')
        ax.set_xlabel('Sigma')
        ax.set_ylabel('Accuracy')
        plt.show()

    elif type_of_attack == 'w':
        ## de continuat cu atacuri de tip white-box
        disable_eager_execution()
        # model_constrained = KerasClassifier(model=model_constrained)#, Having problems with casting
        # #                                     # input_layer=0,
        # #                                     # output_layer=0)

        model_unconstrained = KerasClassifier(model=model_unconstrained)#,model=model_unconstrained, use_logits=False
                                              # input_layer=0,
                                              # output_layer=0)
        attack = FastGradientMethod(estimator=model_unconstrained, eps=0.2)

