from Load_data import DatasetGenerator
from CNN_models import deep_cnn, cnn_trad_pool3
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

DIR = './speech_commands/'
LABELS = 'yes no up down left right on off stop go'.split()
NUM_CLASSES = len(LABELS)

dsGen = DatasetGenerator(label_set=LABELS)
df = dsGen.load_data(DIR)
dsGen.train_test_split(test_size=0.1, random_state=2018)
dsGen.train_validation_split(val_size=0.1, random_state=2018)

BATCH = 30
EPOCHS = 15

model = cnn_trad_pool3()
callbacks = [EarlyStopping(monitor='val_acc', patience=4, verbose=1, mode='max')]
history = model.fit_generator(generator=dsGen.generator(BATCH, mode='train'),
                              steps_per_epoch=int(np.ceil(len(dsGen.train_data) / BATCH)),
                              epochs=EPOCHS,
                              verbose=1,
                              callbacks=callbacks,
                              validation_data=dsGen.generator(BATCH, mode='val'),
                              validation_steps=int(np.ceil(len(dsGen.validation_data) / BATCH)))

y_pred_proba = model.predict_generator(dsGen.generator(BATCH, mode='test'), int(np.ceil(len(dsGen.test_data)/BATCH)), verbose=1)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = dsGen.test_data['label_id'].values
acc_score = accuracy_score(y_true, y_pred)
print(acc_score)

model = deep_cnn()
callbacks = [EarlyStopping(monitor='val_acc', patience=4, verbose=1, mode='max')]
history = model.fit_generator(generator=dsGen.generator(BATCH, mode='train'),
                              steps_per_epoch=int(np.ceil(len(dsGen.train_data) / BATCH)),
                              epochs=EPOCHS,
                              verbose=1,
                              callbacks=callbacks,
                              validation_data=dsGen.generator(BATCH, mode='val'),
                              validation_steps=int(np.ceil(len(dsGen.validation_data) / BATCH)))

y_pred_proba = model.predict_generator(dsGen.generator(BATCH, mode='test'), int(np.ceil(len(dsGen.test_data)/BATCH)), verbose=1)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = dsGen.test_data['label_id'].values
acc_score = accuracy_score(y_true, y_pred)
print(acc_score)


