import os
import numpy as np
import pandas as pd
from glob import glob
import random
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from scipy.io import wavfile
from scipy.signal import stft


class DatasetGenerator():
    def __init__(self, label_set, sample_rate=16000):
        self.label_set = label_set
        self.sample_rate = sample_rate

    def text_to_labels(self, text):
        return self.label_set.index(text)

    def labels_to_text(self, labels):
        return self.label_set[labels]

    def load_data(self, DIR):
        wav_files = glob(os.path.join(DIR, '*/*wav'))
        wav_files = [x.split(sep='\\')[1] + '/' + x.split(sep='\\')[2] for x in wav_files]
        data = []
        for e in wav_files:
            label, name = e.split('/')
            if label in self.label_set:
                label_id = self.text_to_labels(label)
                fle = os.path.join(DIR, e)
                sample = (label, label_id, name, fle)
                data.append(sample)
        dataframe = pd.DataFrame(data, columns=['label', 'label_id', 'user_id', 'wav_file'])
        self.dataframe = dataframe
        return self.dataframe

    def train_test_split(self, test_size, random_state):
        self.train_data, self.test_data = train_test_split(self.dataframe, test_size=test_size, random_state=random_state)

    def train_validation_split(self, val_size, random_state):
        self.train_data, self.validation_data = train_test_split(self.train_data, test_size=val_size, random_state=random_state)

    def read_wav_file(self, x):
        _, wav = wavfile.read(x)
        wav = wav.astype(np.float32)/np.iinfo(np.int16).max
        return wav

    def process_wav_file(self, x, threshold_freq=5500, eps=1e-10):
        wav = self.read_wav_file(x)
        L = self.sample_rate
        if len(wav) > L:
            i = np.random.randint(0, len(wav) - L)
            wav = wav[i:(i + L)]
        elif len(wav) < L:
            rem_len = L - len(wav)
            silence_part = np.random.randint(-100, 100, 16000).astype(np.float32) / np.iinfo(np.int16).max
            j = np.random.randint(0, rem_len)
            silence_part_left = silence_part[0:j]
            silence_part_right = silence_part[j:rem_len]
            wav = np.concatenate([silence_part_left, wav, silence_part_right])
        freqs, times, spec = stft(wav, L, nperseg=400, noverlap=240, nfft=512, padded=False, boundary=None)
        if threshold_freq is not None:
            spec = spec[freqs <= threshold_freq, :]
            freqs = freqs[freqs <= threshold_freq]
        amp = np.log(np.abs(spec) + eps)
        return np.expand_dims(amp, axis=2)

    def generator(self, batch_size, mode):
        while True:
            if mode == 'train':
                df = self.train_data
                ids = random.sample(range(df.shape[0]), df.shape[0])
            elif mode == 'val':
                df = self.validation_data
                ids = list(range(df.shape[0]))
            elif mode == 'test':
                df = self.test_data
                ids = list(range(df.shape[0]))
            else:
                raise ValueError('The mode should be either train, val or test.')

            for start in range(0, len(ids), batch_size):
                X_batch = []
                if mode != 'test':
                    y_batch = []
                end = min(start + batch_size, len(ids))
                i_batch = ids[start:end]
                for i in i_batch:
                    X_batch.append(self.process_wav_file(df.wav_file.values[i]))
                    if mode != 'test':
                        y_batch.append(df.label_id.values[i])
                X_batch = np.array(X_batch)

                if mode != 'test':
                    y_batch = to_categorical(y_batch, num_classes=len(self.label_set))
                    yield (X_batch, y_batch)
                else:
                    yield X_batch