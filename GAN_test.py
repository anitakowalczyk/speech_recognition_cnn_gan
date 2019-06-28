from keras.utils import np_utils
from Generator import build_generator
from Discriminator import build_discriminator
from keras.layers import Input
from keras.optimizers import Adam
from keras.models import Model
import os, cv2
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#load data
PATH = os.getcwd()
data_path = "./resized_spectograms"
data_dir_list = os.listdir(data_path)
img_data_list = []

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img= cv2.imread(data_path + '/' + dataset + '/' + img)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        img_data_list.append(input_img)
img_data = np.array(img_data_list)

#preprocessing
img_data_list = []
for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_flatten = cv2.resize(input_img, (28, 28)).flatten()
        img_data_list.append(input_img_flatten)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data_scaled = preprocessing.scale(img_data)
img_data_scaled = img_data_scaled.reshape(img_data.shape[0], 1, 28, 28)
img_data = img_data_scaled

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype='int64')
labels[0:400] = 0
labels[400:800] = 1
labels[800:] = 2
names = ['yes', 'no', 'up']

batch_size = 10
buffer_size = 1200
epochs = 500

valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))
Y = np_utils.to_categorical(labels, 3)
x, y = shuffle(img_data, Y, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
X_train = X_train.astype("float32")
X_train = (X_train - 127.5) / 127.5

#generator pobiera szum jako wejście i generuje obrazy
generator = build_generator()
z = Input(shape=(100,))
img = generator(z)

#budowa dyskryminatora
optimizer = Adam(0.0002, 0.5)
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
validity = discriminator(img) #dyskryminator pobiera wygenerowane obrazy jako dane wejściowe i określa ich poprawność
combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

for epoch in range(epochs):
    # trenowanie dyskryminatora
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs = X_train[idx]
    noise = np.random.normal(0, 1, (batch_size, 100))
    gen_imgs = generator.predict(noise)
    d_loss_real = discriminator.train_on_batch(imgs, valid)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # trenowanie generatora
    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss = combined.train_on_batch(noise, valid)

    #print("%d | d_loss = %f, acc.= %.2f%%, g_loss = %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
