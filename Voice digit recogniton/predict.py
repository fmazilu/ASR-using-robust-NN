# This file is used to evaluate the models built using train_constraints.py or train_google_dataset.py
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import pathlib
from sklearn.utils import shuffle
from extract_features_construct_dataset import get_file_names_and_labels, compute_mfcc_all_files
import tensorflow as tf
import os
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from train_constraints import customConstraint
from extract_features_construct_dataset import get_norms, get_upper_lipschitz, get_lipschitz_constrained

def main():
    data_dir = pathlib.Path('test_hidden_commands\\')
    save_dir = 'ownTest'

    # Get files
    filenames, labels1 = get_file_names_and_labels(data_dir)

    # shuffle files and labels at the same time
    # filenames, labels1 = shuffle(filenames, labels1)

    # Compute MFCC for all files
    mfcc_own_test = compute_mfcc_all_files(filenames)
    labels = to_categorical(labels1, 10)

    ## Standardize data
    scaler1 = StandardScaler()
    mfcc_own_test = scaler1.fit_transform(mfcc_own_test)
    choose_model = input("[B]aseline or [R]obust baseline: ").lower()
    if choose_model == 'b':
        model = load_model("bin/models/baseline.h5")
    elif choose_model == 'r':
        model = load_model("bin/models_constrained/model_constrained_Rho01_dropout01.h5")
        #### There is a problem loading the model with customConstraint
        # custom_objects={'customConstraint': customConstraint})
    print(model.summary())

    y = np.argmax(model.predict(mfcc_own_test), axis=1)
    print(y)
    print(f'gound truth = {labels1}')
    results = model.evaluate(mfcc_own_test, labels)
    print(f'Test loss: {results[0]} / Test accuracy: {results[1]}')

    norms = get_norms(model)
    print(norms)

    lip = get_upper_lipschitz(norms)
    print("Upper-bound Lipschitz constant: " + str(lip))

    cst = get_lipschitz_constrained(model)
    print("Lipschitz constant for constrained model: " + str(cst))


if __name__ == '__main__':
    main()
