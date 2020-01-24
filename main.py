import numpy as np
import librosa as lbr
import tensorflow.keras.backend as backend
GENRES = ['blues', 'classical', 'jazz', 'metal', 'reggae']
MEL_KWARGS = {
    'n_fft': 2048,
    'hop_length': 2048 // 2,
    'n_mels': 128
}
from sys import stderr, argv
import os
N_LAYERS = 2
EPOCH_COUNT = 50
import sys
from math import pi
from optparse import OptionParser
TRACK_COUNT = 450
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, Activation, TimeDistributed, Convolution1D, MaxPooling1D, BatchNormalization
from sklearn.model_selection import train_test_split

def load_track(filename, enforce_shape=None):
    new_input, sample_rate = lbr.load(filename, mono=True)
    features = lbr.feature.melspectrogram(new_input, **MEL_KWARGS).T
    #features = lbr.power_to_db(features, ref=np.max)
    if enforce_shape is not None:
        if features.shape[0] < enforce_shape[0]:
            delta_shape = (enforce_shape[0] - features.shape[0],
                           enforce_shape[1])
            features = np.append(features, np.zeros(delta_shape), axis=0)
        elif features.shape[0] > enforce_shape[0]:
            features = features[: enforce_shape[0], :]
    features[features == 0] = 1e-6
    return (np.log(features), float(new_input.shape[0]) / sample_rate)
def get_default_shape(dataset_path):
    tmp_features, _ = load_track(os.path.join(dataset_path,
                                              'blues/blues.00000.wav'))
    return tmp_features.shape
def feature_extraction(dataset_path):
    default_shape = get_default_shape(dataset_path)
    x = np.zeros((TRACK_COUNT,) + default_shape, dtype=np.float32)
    y = np.zeros((TRACK_COUNT, len(GENRES)), dtype=np.float32)
    track_paths = {}
    for (genre_index, genre_name) in enumerate(GENRES):
        i = 0
        while i < TRACK_COUNT // len(GENRES):
            file_name = '{}/{}.000{}.wav'.format(genre_name,
                                                genre_name, str(i).zfill(2))
            print('Procesiram fajl ', file_name)
            path = os.path.join(dataset_path, file_name)
            track_index = genre_index * (TRACK_COUNT // len(GENRES)) + i
            x[track_index], _ = load_track(path, default_shape)
            y[track_index, genre_index] = 1
            track_paths[track_index] = os.path.abspath(path)
            i = i + 1
    return (x, y, track_paths)
def train_model(data, model_path):
    x = data['x']
    y = data['y']
    (x_train, x_val, y_train, y_val) = train_test_split(x, y, test_size=0.2,
                                                        random_state=42)
    print('Treniraj')
    n_features = x_train.shape[2]
    input_shape = (None, n_features)
    model_input = Input(input_shape, name='input')
    layer = model_input
    for i in range(N_LAYERS):
        # second convolutional layer names are used by extract_filters.py
        layer = Convolution1D(
            filters=256,
            kernel_size=5,
            name='convolution_' + str(i + 1)
        )(layer)
        layer = BatchNormalization(momentum=0.9)(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling1D(2)(layer)
        layer = Dropout(0.2)(layer)
    layer = TimeDistributed(Dense(len(GENRES)))(layer)
    time_distributed_merge_layer = Lambda(
        function=lambda x: backend.mean(x, axis=1),
        output_shape=lambda shape: (shape[0],) + shape[2:],
        name='output_merged'
    )
    layer = time_distributed_merge_layer(layer)
    layer = Activation('softmax', name='output_realtime')(layer)
    model_output = layer
    model = Model(model_input, model_output)
    #opt = Adam(lr=0.001)
    opt = RMSprop(lr=0.01)
    model.compile(
        #loss='categorical_crossentropy',
        loss= 'binary_crossentropy',
        #loss= 'poisson',
        optimizer=opt,
        metrics=['accuracy']
    )
    print('Treniram')
    model.fit(
        x_train, y_train, batch_size=32, nb_epoch=EPOCH_COUNT,
        validation_data=(x_val, y_val), verbose=1, callbacks=[
            ModelCheckpoint(
                model_path, save_best_only=True, monitor='val_accuracy', verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy', factor=0.5, patience=10, min_delta=0.01,
                verbose=1
            )
        ]
    )
    return model

if __name__ == '__main__':
    parser = OptionParser()
    (x, y, track_paths) = feature_extraction(os.path.join(os.path.dirname(__file__), 'data'))
    data = {'x': x, 'y': y, 'track_paths': track_paths}
   
    train_model(data, os.path.join(os.path.dirname(__file__),
                                           'models/model.h5'))
   