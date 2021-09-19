## pax version of
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/master/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

import math

import jax
import jax.numpy as jnp
import pax
from einops import rearrange
from pax.nn import GroupNorm, LayerNorm


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
        emb = jnp.concatenate(
            (jnp.sin(emb), jnp.cos(emb)),
            axis=-1,
        )
        return emb


class Mish(pax.Module):
    def __call__(self, x):
        return x * jnp.tanh(jax.nn.softplus(x))


class Upsample(pax.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = pax.nn.Conv2DTranspose(dim, dim, 4, 2, "VALID")

    def __call__(self, x):
        return self.conv(x)


class Downsample(pax.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = pax.nn.Conv2D(dim, dim, 3, 2, "SAME")

    def __call__(self, x):
        return self.conv(x)


class PreNorm(pax.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def __call__(self, x):
        x = self.norm(x)
        return self.fn(x)


class Block(pax.Module):
    def __init__(self, dim, dim_out, groups: int = 8):
        super().__init__()
        self.blocks = pax.nn.Sequential(
            pax.nn.Conv2D(dim, dim_out, 3, "SAME"), GroupNorm(groups, dim_out), Mish()
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(pax.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            pax.nn.Sequential(Mish(), pax.nn.Linear(time_emb_dim, dim_out))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = (
            pax.nn.Conv2D(dim, dim_out, 1) if dim != dim_out else lambda x: x
        )

    def __call__(self, x, time_emb):
        h = self.block1(x)

        if self.mlp is not None:
            h = h + self.mlp(time_emb)[:, :, None, None]

        h = self.block2(2)
        return h + self.res_conv(x)


class LinearAttention(pax.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = pax.nn.Conv2D(dim, hidden_dim * 3, 1, with_bias=False)
        self.to_out = pax.nn.Conv2D(hidden_dim, dim, 1)

    def __call__(self, x):
        b, h, w, c = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b h w (qkv heads c) -> qkv b h w heads c", heads=self.heads, qkv=3
        )
        k = jax.nn.softmax(k, axis=-1)
        context = jnp.einsum("bhdn,bhen->bhde", k, v)
        out = jnp.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads (h w) c -> b h w (heads c)", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


### model


class UNet(pax.Module):
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

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = pax.nn.Sequential(
                SinusoidalPosEmbed(dim),
                pax.nn.Linear(dim, dim * 4),
                Mish(),
                pax.nn.Linear(dim * 4, dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        ups, downs = [], []
        num_resolutions = len(in_out)
