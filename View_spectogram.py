import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import stft

fns = ['./speech_commands/off/00b01445_nohash_0.wav',
       './speech_commands/go/00b01445_nohash_0.wav',
       './speech_commands/yes/00f0204f_nohash_0.wav']
SAMPLE_RATE = 16000


def read_wav_file(x):
    _, wav = wavfile.read(x)
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return wav


fig = plt.figure(figsize=(14, 8))
for i, fn in enumerate(fns):
    wav = read_wav_file(fn)
    ax = fig.add_subplot(3, 1, i + 1)
    ax.set_title('Raw wave of ' + fn)
    ax.set_ylabel('Amplitude')
    ax.plot(np.linspace(0, SAMPLE_RATE / len(wav), SAMPLE_RATE), wav)
fig.tight_layout()
plt.show()


def log_spectrogram(wav):
    freqs, times, spec = stft(wav, SAMPLE_RATE, nperseg=400, noverlap=240, nfft=512, padded=False, boundary=None)
    amp = np.log(np.abs(spec) + 1e-10)
    return freqs, times, amp


fig = plt.figure(figsize=(14, 8))
for i, fn in enumerate(fns):
    wav = read_wav_file(fn)
    freqs, times, amp = log_spectrogram(wav)
    ax = fig.add_subplot(3, 1, i + 1)
    ax.imshow(amp, aspect='auto', origin='lower', extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    ax.set_title('Spectrogram of ' + fn)
    ax.set_ylabel('Freqs in Hz')
    ax.set_xlabel('Seconds')
fig.tight_layout()
plt.show()