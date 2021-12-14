from functools import partial

import jax
import jax.numpy as jnp
import pax


class UpsampleNet(pax.Module):
    """Upsampling melspectrogram."""

    def __init__(self, n_mels, num_output_channels):
        super().__init__()
        self.input_conv = pax.Conv1D(n_mels, 512, 1, padding="VALID")
        self.dilated_convs = []
        self.bns = []
        for i in range(5):
            conv = pax.Conv1D(512, 512, 3, rate=2 ** i, padding="VALID")
            self.dilated_convs.append(conv)
            self.bns.append(pax.BatchNorm1D(512, True, True, 0.99))
        self.upsample_conv_1 = pax.Conv1DTranspose(512, 512, 4, stride=4)
        self.upsample_bn1 = pax.BatchNorm1D(512, True, True, 0.99)
        self.upsample_conv_2 = pax.Conv1DTranspose(512, 512, 4, stride=4)
        self.upsample_bn2 = pax.BatchNorm1D(512, True, True, 0.99)
        self.output_conv = pax.Conv1D(512, num_output_channels, 1, padding="VALID")

    def __call__(self, mel):
        x = self.input_conv(mel)

        # Large receptive fields
        for conv, batch_norm in zip(self.dilated_convs, self.bns):
            residual = jax.nn.relu(batch_norm(conv(x)))
            pad = (x.shape[1] - residual.shape[1]) // 2
            x = x[:, pad:-pad] + residual

        # upsample
        x = jax.nn.relu(self.upsample_bn1(self.upsample_conv_1(x)))
        x = jax.nn.relu(self.upsample_bn2(self.upsample_conv_2(x)))

        x = self.output_conv(x)

        # tile x16
        N, L, D = x.shape
        x = jnp.tile(x[:, :, None, :], (1, 1, 16, 1))
        x = jnp.reshape(x, (N, -1, D))

        return x


class WaveGRU(pax.Module):
    def __init__(self, n_mels, hidden_dim, n_mu_bits=8):
        super().__init__()
        self.n_mu_bits = n_mu_bits
        self.hidden_dim = hidden_dim

        self.upsampling = UpsampleNet(n_mels, hidden_dim)
        self.gru = pax.GRU(hidden_dim, hidden_dim)
        self.logits = pax.Linear(hidden_dim, 2 ** n_mu_bits)
        self.embed = pax.Embed(2 ** n_mu_bits, hidden_dim)

    def __call__(self, inputs):
        logmel, wav = inputs
        x = self.upsampling(logmel)
        hx = self.gru.initial_state(x.shape[0])
        wav = self.embed(wav)
        assert x.shape == wav.shape
        x = x + wav
        _, x = pax.scan(self.gru, hx, x, time_major=False)
        x = self.logits(x)
        return x

    def inference(self, logmel, rng_key=None):
        if rng_key is None:
            rng_key = pax.next_rng_key()

        x = jnp.array([2 ** (self.n_mu_bits - 1)], dtype=jnp.int32)
        hx = self.gru.initial_state(1)

        conds = self.upsampling(logmel)

        def loop(prev_state, inputs):
            x, hx, rng_key = prev_state
            rng_key, next_rng_key = jax.random.split(rng_key)

            x = self.embed(x) + inputs
            hx, x = self.gru(hx, x)
            x = self.logits(x)
            x = jax.random.categorical(rng_key, x)
            return (x, hx, next_rng_key), x

        _, x = pax.scan(loop, (x, hx, rng_key), conds, time_major=False)
        return x
