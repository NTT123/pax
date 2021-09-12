import jax
import jax.numpy as jnp
import pax


class UpsamplingNetwork(pax.Module):
    def __init__(self, n_mels, num_output_channels):
        super().__init__()
        self.input_conv = pax.nn.Conv1D(
            n_mels, 512, 3, padding="VALID", with_bias=False
        )
        self.input_bn = pax.nn.BatchNorm1D(512, True, True, 0.99)
        self.dilated_conv_1 = pax.nn.Conv1D(
            512, 512, 2, 1, rate=2, padding="VALID", with_bias=False
        )
        self.dilated_bn_1 = pax.nn.BatchNorm1D(512, True, True, 0.99)
        self.dilated_conv_2 = pax.nn.Conv1D(
            512, 512, 2, 1, rate=4, padding="VALID", with_bias=False
        )
        self.dilated_bn_2 = pax.nn.BatchNorm1D(512, True, True, 0.99)

        self.upsample_conv_1 = pax.nn.Conv1DTranspose(
            512, 512, kernel_shape=1, stride=2, padding="SAME", with_bias=False
        )
        self.upsample_bn_1 = pax.nn.BatchNorm1D(512, True, True, 0.99)
        self.upsample_conv_2 = pax.nn.Conv1DTranspose(
            512, 512, kernel_shape=1, stride=2, padding="SAME", with_bias=False
        )
        self.upsample_bn_2 = pax.nn.BatchNorm1D(512, True, True, 0.99)
        self.upsample_conv_3 = pax.nn.Conv1DTranspose(
            512,
            num_output_channels,
            kernel_shape=1,
            stride=4,
            padding="SAME",
            with_bias=False,
        )
        self.upsample_bn_3 = pax.nn.BatchNorm1D(num_output_channels, True, True, 0.99)

    def __call__(self, mel):
        x = jax.nn.relu(self.input_bn(self.input_conv(mel)))
        res_1 = jax.nn.relu(self.dilated_bn_1(self.dilated_conv_1(x)))
        x = x[:, 1:-1] + res_1
        res_2 = jax.nn.relu(self.dilated_bn_2(self.dilated_conv_2(x)))
        x = x[:, 2:-2] + res_2

        x = jax.nn.relu(self.upsample_bn_1(self.upsample_conv_1(x)))
        x = jax.nn.relu(self.upsample_bn_2(self.upsample_conv_2(x)))
        x = jax.nn.relu(self.upsample_bn_3(self.upsample_conv_3(x)))

        # tile x16
        N, L, D = x.shape
        x = jnp.tile(x[:, :, None, :], (1, 1, 16, 1))
        x = jnp.reshape(x, (N, -1, D))

        return x


class WaveGRU(pax.Module):
    def __init__(self, n_mels, hidden_dim, n_mu_bits=8):
        super().__init__()

        self.upsampling = UpsamplingNetwork(n_mels, hidden_dim)
        self.gru = pax.nn.GRU(hidden_dim, hidden_dim)
        self.O1 = pax.nn.Linear(hidden_dim, hidden_dim, name="O1")
        self.O2 = pax.nn.Linear(hidden_dim, 2 ** n_mu_bits, name="O2")
        self.hidden_dim = hidden_dim
        self.embed = pax.nn.Embed(2 ** n_mu_bits, hidden_dim)
        self.n_mu_bits = n_mu_bits

    def __call__(self, inputs):
        logmel, wav = inputs
        x = self.upsampling(logmel)
        hx = self.gru.initial_state(x.shape[0])
        wav = self.embed(wav)
        assert x.shape == wav.shape
        x = x + wav
        _, x = pax.utils.scan(self.gru, hx, x, time_major=False)
        x = jax.nn.relu(self.O1(x))
        x = self.O2(x)
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
            x = self.O2(jax.nn.relu(self.O1(x)))
            x = jax.random.categorical(rng_key, x)
            return (x, hx, next_rng_key), x

        _, x = pax.utils.scan(loop, (x, hx, rng_key), conds, time_major=False)
        return x
