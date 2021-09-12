from functools import partial

import jax
import jax.numpy as jnp
import librosa
import opax
import pax
import soundfile
from tqdm.auto import tqdm

from data_loader import data_loader
from model import WaveGRU


def loss_fn(params: WaveGRU, model: WaveGRU, inputs) -> pax.utils.LossFnOutput:
    model = model.update(params)
    logmel, wav = inputs
    input_wav = wav[:, :-1]
    target_wav = wav[:, 1:]
    logits = model((logmel, input_wav))
    log_pr = jax.nn.log_softmax(logits, axis=-1)
    target_wave = jax.nn.one_hot(target_wav, num_classes=logits.shape[-1])
    log_pr = jnp.sum(log_pr * target_wave, axis=-1)
    loss = -jnp.mean(log_pr)
    return loss, (loss, model)


def generate_test_sample(step, test_logmel, wave_gru, length, sample_rate, mu):
    generated_mu = wave_gru.eval().inference(test_logmel[None, :length, :])
    generated_mu = jax.device_get(generated_mu)
    synthesized_clip = librosa.mu_expand(generated_mu[0] - 128, mu=mu, quantize=True)
    soundfile.write(
        f"/tmp/wave_gru_sample_{step:05d}.wav",
        synthesized_clip,
        samplerate=sample_rate,
    )


def train(
    hidden_dim: int = 512,
    num_training_steps: int = 5_000,
    batch_size: int = 128,
    learning_rate: float = 5e-4,
    sample_rate: int = 16_000,
    max_global_norm: float = 1.0,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    fmin=0,
    fmax=8000,
    seq_len=2 ** 10,
    n_mu_bits=8,
    log_freq: int = 1000,
):
    mu = 2 ** n_mu_bits - 1
    n_frames = seq_len // hop_length
    wave_gru = WaveGRU(n_mels, hidden_dim)
    print(wave_gru.summary())

    optimizer = opax.chain(
        opax.clip_by_global_norm(max_global_norm),
        opax.adam(learning_rate),
    )(wave_gru.parameters())

    split_loader = partial(
        data_loader,
        batch_size=batch_size,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        sample_rate=sample_rate,
        mu=mu,
        n_frames=n_frames,
        fmin=fmin,
        fmax=fmax,
    )
    data_iter = split_loader(split="train")
    test_iter = split_loader(split="test")
    test_logmel, _ = next(test_iter)

    update_fn = jax.jit(pax.utils.build_update_fn(loss_fn))
    total_loss = 0.0
    tr = tqdm(range(1, 1 + num_training_steps))
    for step in tr:
        batch = next(data_iter)
        loss, wave_gru, optimizer = update_fn(wave_gru, optimizer, batch)
        total_loss = total_loss + loss

        if step % log_freq == 0:
            loss = total_loss / log_freq
            tr.write(f"[step {step}]  train loss {loss:.3f}")
            total_loss = 0.0
            generate_test_sample(step, test_logmel, wave_gru, 200, sample_rate, mu)

    return wave_gru


if __name__ == "__main__":
    import fire

    fire.Fire(train)