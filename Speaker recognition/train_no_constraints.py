# This file is used to train models that are unconstrained using Google's Speech Commands Data Set
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, LSTM, Conv1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import datetime
from sklearn.preprocessing import StandardScaler
from extract_features_construct_dataset import get_norms, get_upper_lipschitz
import matplotlib.pyplot as plt
import seaborn as sn

# Loading the datasets
path = 'RoDigits_splitV2/'
train_data = np.load(path + "train_data.npy", allow_pickle=True)
train_label = np.load(path + "train_label.npy")
train_label = to_categorical(train_label, 20)
val_label = np.load(path + "dev_label.npy")
val_label = to_categorical(val_label, 20)
val_data = np.load(path + "dev_data.npy", allow_pickle=True)
test_data = np.load(path + "test_data.npy", allow_pickle=True)
test_label1 = np.load(path + "test_label.npy")
test_label = to_categorical(test_label1, 20)
print(train_data.shape)

# Standardizing the data
all_data = np.concatenate((train_data, val_data, test_data), axis=0)
scaler1 = StandardScaler()
all_data = scaler1.fit_transform(all_data)

train_data = all_data[:train_data.shape[0]]
val_data = all_data[train_data.shape[0]:train_data.shape[0] + val_data.shape[0]]
test_data = all_data[train_data.shape[0] + val_data.shape[0]:]
print(train_data.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_label))
test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))

train_dataset = train_dataset.shuffle(2000, reshuffle_each_iteration=False).batch(64)
val_dataset = val_dataset.shuffle(1000, reshuffle_each_iteration=False).batch(64)


def tensorboard_callback():
    logdir= 'logs/log' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    return tensorboard_callback


def get_model():
    inp = Input((2020,))
    hdn = Dense(1024, activation='relu')(inp)
    # hdn = BatchNormalization()(hdn)
    # hdn = Dropout(0.5)(hdn)

    hdn = Dense(512, activation='relu')(hdn)
    # hdn = BatchNormalization()(hdn)
    # hdn = Dropout(0.4)(hdn)

    hdn = Dense(256, activation='relu')(hdn)
    # hdn = BatchNormalization()(hdn)
    # hdn = Dropout(0.4)(hdn)

    hdn = Dense(128, activation='relu')(hdn)
    # hdn = BatchNormalization()(hdn)

    hdn = Dense(64, activation='relu')(hdn)
    # hdn = BatchNormalization()(hdn)

    out = Dense(20, activation='softmax')(hdn)

    model = Model(inputs=inp, outputs=out)
    return model


def main():
    # Model Training
    model = get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    model.fit(train_dataset, epochs=10000, validation_data=val_dataset, verbose=2,
              callbacks=[tensorboard_callback(),
                         EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=False),
                         ModelCheckpoint('bin/models/baseline_splitV2.h5', save_best_only=True, verbose=1)])
    model = load_model("bin/models/baseline_splitV2.h5")
    print(model.summary())
    model_norms = get_norms(model)
    lip = get_upper_lipschitz(model_norms)
    print(f"Upper Lipschitz constant for non-constrained model: {lip}")
    y = np.argmax(model.predict(test_data), axis=1)
    results = model.evaluate(test_data, test_label)
    print(f'Test loss: {results[0]} / Test accuracy: {results[1]}')


if __name__ == '__main__':
    main()
