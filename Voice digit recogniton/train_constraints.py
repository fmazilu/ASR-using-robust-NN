# This file is used to train models that are constrained using Google's Speech Commands Data Set
import tensorflow as tf
import numpy as np
from tensorflow.keras.constraints import Constraint, NonNeg
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, LSTM, Conv1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CategoricalCrossentropy
import datetime
from sklearn.preprocessing import StandardScaler
from extract_features_construct_dataset import get_lipschitz_constrained
from Constraints import customConstraint, norm_constraint, norm_constraint_FISTA, simple_norm_constraint

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

train_dataset = train_dataset.shuffle(880, reshuffle_each_iteration=False).batch(512)
val_dataset = val_dataset.shuffle(880, reshuffle_each_iteration=False).batch(512)


def tensorboard_callback():
    logdir= 'logs/log_constrained' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    return tensorboard_callback


# Callback class for monitoring the Lipschitz constant
class lip_stats_callback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lip_cst = get_lipschitz_constrained(model)
        for layer in self.model.layers:
            if 'dense' in layer.name:
                w = layer.get_weights()[0]
                norm = np.linalg.norm(w, ord=2)
                print(f"The norm for layer {layer} is : {norm}")
        print(f'The Lipschitz constant on epoch {epoch} is {lip_cst}')


def get_model():
    m = 5
    rho = 20
    inp = Input((880,))
    hdn = Dense(1024, activation='relu', kernel_constraint=NonNeg())(inp)
    hdn = BatchNormalization()(hdn)
    hdn = Dropout(0.1)(hdn)

    hdn = Dense(512, activation='relu', kernel_constraint=NonNeg())(hdn)
    hdn = BatchNormalization()(hdn)
    hdn = Dropout(0.1)(hdn)

    hdn = Dense(256, activation='relu', kernel_constraint=NonNeg())(hdn)
    hdn = BatchNormalization()(hdn)
    hdn = Dropout(0.1)(hdn)

    hdn = Dense(128, activation='relu', kernel_constraint=NonNeg())(hdn)
    hdn = BatchNormalization()(hdn)

    hdn = Dense(64, activation='relu', kernel_constraint=NonNeg())(hdn)
    hdn = BatchNormalization()(hdn)

    out = Dense(10, activation='softmax', kernel_constraint=NonNeg())(hdn)

    model = Model(inputs=inp, outputs=out)
    return model


if __name__ == '__main__':
    # Model Training
    model = get_model()
    model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=['accuracy'])
    print(model.summary())
    # model.load_weights('bin/models_constrained/model_1layer.h5')
    model.fit(train_dataset, epochs=10000, validation_data=val_dataset, verbose=2,
              callbacks=[tensorboard_callback(),
                         EarlyStopping(monitor="val_loss", patience=6000, restore_best_weights=False),
                         # norm_constraint_FISTA(rho=5, nit=2),
                         # norm_constraint(rho=10),     ## Read more about layer indices
                         simple_norm_constraint(rho=0.1, affected_layers_indices=[]),
                         lip_stats_callback(),
                         ModelCheckpoint('bin/models_constrained/model_constrained_Rho01_dropout01.h5',
                                         save_best_only=True, verbose=1)])

    model = load_model("bin/models_constrained/model_constrained_Rho01_dropout01.h5")
    print(model.summary())
    y = np.argmax(model.predict(test_data), axis=1)
    results = model.evaluate(test_data, test_label)
    print(f'Test loss: {results[0]} / Test accuracy: {results[1]}')

    # Plotting The confusion matrix
    # conf_matrix = tf.math.confusion_matrix(test_label1, y)
    # print(conf_matrix[0])
    # ax = sn.heatmap(conf_matrix)
    # ax.legend()
    # ax.set_title("Confusion Matrix")
    # plt.show()
