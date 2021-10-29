from functools import partial

import jax
import jax.numpy as jnp
import pax


class UpsamplingNetwork(pax.Module):
    def __init__(self, n_mels, num_output_channels):
        super().__init__()
        self.input_conv = pax.nn.Conv1D(n_mels, 512, 3, padding="VALID", with_bias=True)

        dilated_conv = partial(pax.nn.Conv1D, 512, 512, 2, padding="VALID")
        self.dilated_conv_1 = dilated_conv(rate=1)
        self.dilated_conv_2 = dilated_conv(rate=2)
        self.dilated_conv_4 = dilated_conv(rate=4)

        conv1d_transpose = partial(pax.nn.Conv1DTranspose, 512, 512, padding="SAME")
        self.upsample_conv_1 = conv1d_transpose(kernel_shape=4, stride=4)
        self.upsample_conv_2 = conv1d_transpose(kernel_shape=2, stride=2)
        self.upsample_conv_3 = conv1d_transpose(kernel_shape=2, stride=2)
        self.output_conv = pax.nn.Conv1D(512, num_output_channels, 1, padding="VALID")

    def __call__(self, mel):
        x = self.input_conv(mel)
        res_1 = jax.nn.relu(self.dilated_conv_1(x))
        x = x[:, 1:] + res_1
        res_2 = jax.nn.relu(self.dilated_conv_2(x))
        x = x[:, 2:] + res_2
        res_3 = jax.nn.relu(self.dilated_conv_3(x))
        x = x[:, 4:] + res_3

        x = jax.nn.relu(self.upsample_conv_1(x))
        x = jax.nn.relu(self.upsample_conv_2(x))
        x = jax.nn.relu(self.upsample_conv_3(x))

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

        self.upsampling = UpsamplingNetwork(n_mels, hidden_dim)
        self.gru = pax.nn.GRU(hidden_dim, hidden_dim)
        self.logits = pax.nn.Linear(hidden_dim, 2 ** n_mu_bits)
        self.embed = pax.nn.Embed(2 ** n_mu_bits, hidden_dim)

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
