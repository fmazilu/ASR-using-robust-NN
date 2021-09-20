import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, LSTM, Conv1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sn

# Loading the datasets
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

train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_label))
test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))

train_dataset = train_dataset.shuffle(880, reshuffle_each_iteration=False).batch(64)
val_dataset = val_dataset.shuffle(880, reshuffle_each_iteration=False).batch(64)


def tensorboard_callback():
    logdir= 'logs/log' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    return tensorboard_callback


def get_model():
    inp = Input((880,))
    hdn = Dense(256, activation='relu')(inp)
    hdn = Dropout(0.5)(hdn)
    hdn = Dense(128, activation='relu')(hdn)
    hdn = Dropout(0.5)(hdn)
    hdn = Dense(64, activation='relu')(hdn)
    out = Dense(10, activation='softmax')(hdn)

    model = Model(inputs=inp, outputs=out)
    return model


if __name__ == '__main__':
    #### Model Training
    model = get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    model.fit(train_dataset, epochs=600, validation_data=val_dataset, verbose=2,
              callbacks=[tensorboard_callback(),
                         EarlyStopping(monitor="val_loss", patience=150, restore_best_weights=False),
                         ModelCheckpoint('bin/models/model_dropout0.5.h5', save_best_only=True, verbose=1)])
              # class_weight=class_weight)
    y = np.argmax(model.predict(test_data), axis=1)
    results = model.evaluate(test_data, test_label)
    print(f'Test loss: {results[0]} / Test accuracy: {results[1]}')

    #### Plotting The confusion matrix
    conf_matrix = tf.math.confusion_matrix(test_label1, y)
    print(conf_matrix)
    ax = sn.heatmap(conf_matrix)
    ax.legend()
    ax.set_title("Confusion Matrix")
    plt.show()
