from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers import Input, LeakyReLU, BatchNormalization, Reshape
import numpy as np

def build_generator():
    img_shape = (1, 28, 28,)
    noise_shape = 100
    model = Sequential()
    model.add(Dense(256, input_dim=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    print('GENERATOR MODEL')
    model.summary()
    noise = Input(shape=(noise_shape,))
    img = model(noise)
    return Model(noise, img)