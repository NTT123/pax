# Source:
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/master/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

import math
from functools import partial
from inspect import isfunction
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import pax
from einops import rearrange
from pax import GroupNorm, LayerNorm


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(pax.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def __call__(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmbed(pax.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, x):
        half_dim = self.dim // 2
        emb = math.log(10_000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(0, half_dim) * (-emb))
        emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb


class Mish(pax.Module):
    def __call__(self, x):
        return x * jnp.tanh(jax.nn.softplus(x))


class Upsample(pax.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = pax.Conv2DTranspose(dim, dim, 4, 2, padding="SAME")

    def __call__(self, x):
        return self.conv(x)


class Downsample(pax.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = pax.Conv2D(dim, dim, 3, 2, padding="SAME")

    def __call__(self, x):
        return self.conv(x)


class PreNorm(pax.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim, -1, True, True)

    def __call__(self, x):
        x = self.norm(x)
        return self.fn(x)


class Block(pax.Module):
    def __init__(self, dim, dim_out, groups: int = 8):
        super().__init__()
        self.blocks = pax.Sequential(
            pax.Conv2D(dim, dim_out, 3, padding="SAME"),
            GroupNorm(groups, dim_out),
            Mish(),
        )

    def __call__(self, x):
        return self.blocks(x)


class ResnetBlock(pax.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            pax.Sequential(Mish(), pax.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        if dim != dim_out:
            self.res_conv = pax.Conv2D(dim, dim_out, 1)
        else:
            self.res_conv = pax.Identity()

    def __call__(self, x, time_emb):
        h = self.block1(x)

        if exists(self.mlp):
            h = h + self.mlp(time_emb)[:, None, None, :]

        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(pax.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = pax.Conv2D(dim, hidden_dim * 3, 1, with_bias=False)
        self.to_out = pax.Conv2D(hidden_dim, dim, 1)

    def __call__(self, x):
        b, h, w, c = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b h w (qkv heads c) -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = jax.nn.softmax(k, axis=-1)
        context = jnp.einsum("bhdn,bhen->bhde", k, v)
        out = jnp.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b h w (heads c)", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


### model


class UNet(pax.Module):
    ups: List[List[pax.Module]]
    downs: List[List[pax.Module]]

    def __init__(
        self,
        dim,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        groups=8,
        channels=3,
        with_time_emb=True,
    ):
        super().__init__()
        self.channels = channels

        dims = [channels] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = pax.Sequential(
                SinusoidalPosEmbed(dim),
                pax.Linear(dim, dim * 4),
                Mish(),
                pax.Linear(dim * 4, dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = []
        self.ups = []
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                [
                    ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim, groups=groups),
                    ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim, groups=groups),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Downsample(dim_out) if not is_last else pax.Identity(),
                ]
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=time_dim, groups=groups
        )
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=time_dim, groups=groups
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                [
                    ResnetBlock(
                        dim_out * 2, dim_in, time_emb_dim=time_dim, groups=groups
                    ),
                    ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=groups),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Upsample(dim_in) if not is_last else pax.Identity(),
                ]
            )

        out_dim = default(out_dim, channels)

        self.final_conv = pax.Sequential(
            Block(dim, dim, groups=groups),
            pax.Conv2D(dim, out_dim, 1),
        )

    def __call__(self, x, time):
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        h = []
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = jnp.concatenate((x, h.pop()), axis=-1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = jnp.take_along_axis(a, t, axis=-1)
    out = jnp.reshape(out, (b,) + (1,) * (len(x_shape) - 1))
    return out


def noise_like(rng_key, shape, repeat=False):
    repeat_noise = lambda: jnp.tile(
        jax.random.normal(rng_key, (1, *shape[1:])),
        (shape[0],) + (1,) * (len(shape) - 1),
    )
    noise = lambda: jax.random.normal(rng_key, shape)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


class GaussianDiffusion(pax.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels=3,
        timesteps=1000,
        loss_type="l1",
        betas=None,
        random_seed=42,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.rng_seq = pax.RngSeq(random_seed)

        if exists(betas):
            betas = jnp.array(betas)
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0
        # at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(
            np.maximum(posterior_variance, 1e-20)
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon = jnp.clip(x_recon, -1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, rng_key, clip_denoised=True, repeat_noise=False):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised
        )
        noise = noise_like(rng_key, x.shape, repeat_noise)
        # no noise when t == 0
        nonzero_mask = jnp.reshape(
            (1 - (t == 0).astype(jnp.float32)),
            (b,) + (1,) * (len(x.shape) - 1),
        )

        return model_mean + nonzero_mask * jnp.exp(0.5 * model_log_variance) * noise

    def p_sample_loop(self, shape, rng_key):
        b = shape[0]
        rng_key_, rng_key = jax.random.split(rng_key)
        img = jax.random.normal(rng_key_, shape)

        i_s = jnp.flip(jnp.arange(0, self.num_timesteps, dtype=jnp.int32))
        rng_keys = jax.random.split(rng_key, self.num_timesteps)

        def loop_fn(img, inputs):
            i, rng_key = inputs
            img = self.p_sample(img, jnp.full((b,), i, dtype=jnp.int32), rng_key)
            return img, None

        img, _ = pax.scan(loop_fn, img, (i_s, rng_keys))
        return img

    @partial(jax.jit, static_argnums=[1, 2])
    def sample(self, batch_size=16, random_seed=42):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(
            (batch_size, image_size, image_size, channels),
            jax.random.PRNGKey(random_seed),
        )

    def q_sample(self, x_start, t, noise=None):
        noise = default(
            noise,
            lambda: jax.random.normal(
                self.rng_seq.next_rng_key(), x_start.shape, x_start.dtype
            ),
        )

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None):
        noise = default(
            noise,
            lambda: jax.random.normal(
                self.rng_seq.next_rng_key(), x_start.shape, x_start.dtype
            ),
        )

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t)

        if self.loss_type == "l1":
            loss = jnp.mean(jnp.abs(noise - x_recon))
        elif self.loss_type == "l2":
            loss = jnp.mean(jnp.square(noise - x_recon))
        else:
            raise NotImplementedError()

        return loss

    def __call__(self, x, *args, **kwargs):
        b, h, w, c = x.shape
        img_size = self.image_size
        assert (
            h == img_size and w == img_size
        ), f"height and width of image must be {img_size}"

        t = jax.random.randint(
            self.rng_seq.next_rng_key(),
            (b,),
            minval=0,
            maxval=self.num_timesteps,
            dtype=jnp.int32,
        )
        return self.p_losses(x, t, *args, **kwargs)
