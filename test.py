import tensorflow as tf

import sys
import numpy as np
from math import pi
import os
TRACK_COUNT = 50
import librosa as lbr
import tensorflow.keras.backend as backend
GENRES = ['blues', 'classical', 'jazz', 'metal', 'reggae']
MEL_KWARGS = {
    'n_fft': 2048,
    'hop_length': 2048 // 2,
    'n_mels': 128
}
new_model = tf.keras.models.load_model('models/model.h5')
# Show the model architecture
new_model.summary()

def load_track(filename, enforce_shape=None):
    new_input, sample_rate = lbr.load(filename, mono=True)
    features = lbr.feature.melspectrogram(new_input, **MEL_KWARGS).T
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
            print('Procesiram', file_name)
            path = os.path.join(dataset_path, file_name)
            track_index = genre_index * (TRACK_COUNT // len(GENRES)) + i
            x[track_index], _ = load_track(path, default_shape)
            y[track_index, genre_index] = 1
            track_paths[track_index] = os.path.abspath(path)
            i = i + 1
    return (x, y, track_paths)
(x, y, track_paths) = feature_extraction(os.path.join(os.path.dirname(__file__),'dataval'))
data = {'x': x, 'y': y, 'track_paths': track_paths}
x = data['x']
y = data['y']
loss, acc = new_model.evaluate(x,y)
print('Accuracy: {:5.2f}%'.format(100*acc))