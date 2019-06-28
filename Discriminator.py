from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers import Input, LeakyReLU, Flatten

def build_discriminator():
    img_shape = (1, 28, 28)
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    print('DISCRIMINATOR MODEL')
    model.summary()
    img = Input(shape=img_shape)
    validity = model(img)
    return Model(img, validity)