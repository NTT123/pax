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


def loss_fn(model: WaveGRU, inputs):
    logmel, wav = inputs
    input_wav = wav[:, :-1]
    target_wav = wav[:, 1:]
    model, logits = pax.purecall(model, (logmel, input_wav))
    log_pr = jax.nn.log_softmax(logits, axis=-1)
    target_wave = jax.nn.one_hot(target_wav, num_classes=logits.shape[-1])
    log_pr = jnp.sum(log_pr * target_wave, axis=-1)
    loss = -jnp.mean(log_pr)
    return loss, (loss, model)


def generate_test_sample(step, test_logmel, wave_gru, length, sample_rate, mu):
    generated_mu = wave_gru.eval().inference(test_logmel[None, :length, :])
    generated_mu = jax.device_get(generated_mu)
    synthesized_clip = librosa.mu_expand(
        generated_mu[0] - mu // 2, mu=mu, quantize=True
    )
    file_name = f"/tmp/wave_gru_sample_{step:05d}.wav"
    soundfile.write(
        file_name,
        synthesized_clip,
        samplerate=sample_rate,
    )
    return file_name


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
    random_seed=42,
):
    pax.seed_rng_key(random_seed)
    mu = 2 ** n_mu_bits - 1
    n_frames = seq_len // hop_length
    wave_gru = WaveGRU(n_mels, hidden_dim)
    print(wave_gru.summary())

    optimizer = opax.chain(
        opax.clip_by_global_norm(max_global_norm),
        opax.adam(learning_rate),
    ).init(wave_gru.parameters())

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
        wave_gru, optimizer, loss = update_fn(wave_gru, optimizer, batch)
        total_loss = total_loss + loss

        if step % log_freq == 0:
            loss = total_loss / log_freq
            total_loss = 0.0
            file_name = generate_test_sample(
                step, test_logmel, wave_gru, 1000, sample_rate, mu
            )
            tr.write(
                f"[step {step}]  train loss {loss:.3f}  synthesized clip {file_name}"
            )


if __name__ == "__main__":
    import fire

    fire.Fire(train)
