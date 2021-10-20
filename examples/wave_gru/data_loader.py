import os
import random

import librosa
import numpy as np


def data_loader(
    batch_size: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    sample_rate: int,
    fmin: int,
    fmax: int,
    mu: int,
    n_frames: int,
    split="train",
):
    if not os.path.exists("/tmp/wave_gru_clip.wav"):
        os.system("bash /tmp/prepare_clip.sh")

    wav, _ = librosa.load("/tmp/wave_gru_clip.wav", sr=sample_rate)

    L = len(wav) * 9 // 10
    if split == "train":
        wav = wav[:L]
    else:
        wav = wav[L:]

    mel = librosa.feature.melspectrogram(
        n_mels=n_mels,
        y=wav,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
        center=False,
    )

    mel = mel.T

    logmel = np.log(1e-3 + mel)
    mu_wav = librosa.mu_compress(wav, mu=mu, quantize=True) + mu // 2

    if split == "test":
        yield (logmel, mu_wav)
        return

    batch = []
    while True:
        left = random.randint(4, logmel.shape[0] - n_frames - 8)
        right = left + n_frames
        cond = logmel[(left - 3) : (right + 3)]  # padding purposes
        x = mu_wav[left * hop_length : right * hop_length + 1]
        batch.append((cond, x))
        if len(batch) == batch_size:
            conds, xs = zip(*batch)
            conds = np.array(conds)
            xs = np.array(xs, dtype=np.int16)
            yield (conds, xs)
            batch = []
