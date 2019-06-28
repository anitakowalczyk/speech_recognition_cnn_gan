from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Sequential


def cnn_trad_pool3():
    convNN = Sequential()
    # 32 filtry o rozmiarze 3x3, krok=1, padding zachowauje wymiary przestrzenne woluminu
    # input_shape = rozmiar spektrogramu, rozszerzamy go na tablicę 3D, aby można go było używać w CNN
    convNN.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', input_shape=(177, 98, 1), activation='relu'))
    convNN.add(MaxPooling2D(pool_size=(3, 3)))
    convNN.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
    convNN.add(MaxPooling2D(pool_size=(3, 3)))
    convNN.add(Flatten())
    convNN.add(Dense(10, activation='softmax'))
    convNN.summary()
    convNN.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return convNN


def deep_cnn():
    convNN = Sequential()
    convNN.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=1, input_shape=(177, 98, 1)))
    convNN.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    convNN.add(BatchNormalization())
    convNN.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=1))
    convNN.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    convNN.add(BatchNormalization())
    convNN.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=1))
    convNN.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    convNN.add(BatchNormalization())
    convNN.add(Flatten())
    convNN.add(Dense(64, activation='relu'))
    convNN.add(BatchNormalization())
    convNN.add(Dropout(0.2))
    convNN.add(Dense(10, activation='softmax'))
    convNN.summary()
    convNN.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return convNN