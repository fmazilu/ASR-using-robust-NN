# This file is used to train models that are unconstrained using Google's Speech Commands Data Set
import keras.utils.data_utils
import tensorflow as tf
import numpy as np
import librosa
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.models import load_model
import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sn
import pathlib
from sklearn.utils import shuffle
from extract_features_construct_dataset import get_file_names_and_labels
from scipy.io import wavfile


# Load one single audio file
def load_audio(file_path):
    sampling_rate, raw_w = wavfile.read(file_path)
    # print(raw_w)
    return raw_w


# Load raw audio data
def load_dataset(filenames):
    dataset = np.zeros((len(filenames), 22050))
    for index, file in enumerate(filenames):
        audio = load_audio(file)
        # print(audio.shape)
        audio = np.array(audio)
        if audio.shape[0] > 22050:
            audio = audio[:22050]
        else:
            audio = np.pad(audio, (0, 22050 - audio.shape[0]))
        dataset[index] = audio
    return dataset


class generator(keras.utils.data_utils.Sequence):
    def __init__(self, filenames, labels, batch_size):
        self.filenames = filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        dataset = np.zeros((len(batch_x), 22050))

        for index, file_name in enumerate(batch_x):
            audio = np.array(load_audio(file_name))
            # print(audio.shape)
            audio = np.array(audio)
            if audio.shape[0] > 22050:
                audio = audio[:22050]
            else:
                audio = np.pad(audio, (0, 22050 - audio.shape[0]))
            dataset[index] = audio

        scaler1 = StandardScaler()
        dataset = scaler1.fit_transform(dataset)

        return dataset, np.array(batch_y)


def tensorboard_callback():
    logdir = 'logs/log' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    return tensorboard_callback


def get_model():
    inp = Input((22050,))
    hdn = Dense(4096, activation='relu')(inp)
    hdn = BatchNormalization()(hdn)
    hdn = Dropout(0.8)(hdn)

    hdn = Dense(2048, activation='relu')(hdn)
    hdn = BatchNormalization()(hdn)
    hdn = Dropout(0.6)(hdn)

    hdn = Dense(1024, activation='relu')(hdn)
    hdn = BatchNormalization()(hdn)
    hdn = Dropout(0.6)(hdn)

    hdn = Dense(512, activation='relu')(hdn)
    hdn = BatchNormalization()(hdn)
    hdn = Dropout(0.5)(hdn)

    hdn = Dense(256, activation='relu')(hdn)
    hdn = BatchNormalization()(hdn)
    hdn = Dropout(0.5)(hdn)

    hdn = Dense(64, activation='relu')(hdn)
    hdn = BatchNormalization()(hdn)
    hdn = Dropout(0.5)(hdn)

    out = Dense(10, activation='softmax')(hdn)

    model = Model(inputs=inp, outputs=out)
    return model


def main():
    data_dir = pathlib.Path('data\\')
    filenames, labels = get_file_names_and_labels(data_dir)
    filenames, labels = shuffle(filenames, labels)
    print("1")

    filenames_train = filenames[:int(int(len(filenames)) * 0.7)]
    filenames_dev = filenames[int(int(len(filenames)) * 0.7): int(int(len(filenames)) * 0.9)]
    filenames_test = filenames[-int(int(len(filenames)) * 0.1):]
    labels_train = labels[:int(int(labels.shape[0]) * 0.7)]
    labels_dev = labels[int(int(labels.shape[0]) * 0.7): int(int(labels.shape[0]) * 0.9)]
    labels_test = labels[-int(int(labels.shape[0]) * 0.1):]
    labels_test1 = to_categorical(labels_test, 10)
    test_data = load_dataset(filenames_test)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, labels_test))

    batch_size = 64
    training_generator = generator(filenames_train, to_categorical(labels_train, 10), batch_size)
    validation_generator = generator(filenames_dev, to_categorical(labels_dev, 10), batch_size)

    # Model Training
    model = get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    model.fit_generator(generator=training_generator, steps_per_epoch=int(len(filenames_train) // batch_size), epochs=10000, validation_data=validation_generator, verbose=2,
                        callbacks=[tensorboard_callback(),
                        EarlyStopping(monitor="val_loss", patience=3000, restore_best_weights=False),
                        ModelCheckpoint('bin/models/TEST.h5', save_best_only=True, verbose=1)])
    model = load_model("bin/models/TEST.h5")
    print(model.summary())
    y = np.argmax(model.predict(test_data), axis=1)
    results = model.evaluate(test_data, labels_test1)
    print(f'Test loss: {results[0]} / Test accuracy: {results[1]}')

    # Plotting The confusion matrix
    conf_matrix = tf.math.confusion_matrix(labels_test, y)
    print(conf_matrix)
    ax = sn.heatmap(conf_matrix)
    ax.legend()
    ax.set_title("Confusion Matrix")
    plt.show()


if __name__ == '__main__':
    main()
