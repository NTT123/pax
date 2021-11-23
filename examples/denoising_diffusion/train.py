import math

import fire
import jax
import jax.numpy as jnp
import numpy as np
import opax
import pax
import tensorflow as tf
from PIL import Image
from tqdm.auto import tqdm

from data_loader import load_celeb_a
from model import GaussianDiffusion, UNet


def make_image_grid(images, padding=2):
    """Place images in a square grid."""
    n = images.shape[0]
    size = int(math.sqrt(n))
    assert size * size == n, "expecting a square grid"
    img = images[0]

    H = img.shape[0] * size + padding * (size + 1)
    W = img.shape[1] * size + padding * (size + 1)
    out = np.zeros((H, W, img.shape[-1]), dtype=img.dtype)
    for i in range(n):
        x = i % size
        y = i // size
        xstart = x * (img.shape[0] + padding) + padding
        xend = xstart + img.shape[0]
        ystart = y * (img.shape[1] + padding) + padding
        yend = ystart + img.shape[1]
        out[xstart:xend, ystart:yend, :] = images[i]
    return out


def train(
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    num_training_steps: int = 10_000,
    log_freq: int = 1000,
    image_size: int = 64,
    random_seed: int = 42,
):

    pax.seed_rng_key(random_seed)

    model = UNet(dim=64, dim_mults=(1, 2, 4, 8))

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,
        loss_type="l1",  # L1 or L2
    )

    dataset = load_celeb_a()

    dataloader = (
        dataset.repeat()
        .shuffle(batch_size * 100)
        .batch(batch_size)
        .take(num_training_steps)
        .prefetch(tf.data.AUTOTUNE)
    )

    def loss_fn(model, inputs):
        model, loss = pax.purecall(model, inputs)
        return loss, (loss, model)

    update_fn = pax.utils.build_update_fn(loss_fn)
    fast_update_fn = jax.jit(update_fn)

    optimizer = opax.adam(learning_rate)(diffusion.parameters())

    total_loss = 0.0
    tr = tqdm(dataloader)
    for step, batch in enumerate(tr, 1):
        batch = jax.tree_map(lambda x: x.numpy(), batch)
        diffusion, optimizer, loss = fast_update_fn(diffusion, optimizer, batch)
        total_loss = total_loss + loss

        if step % log_freq == 0:
            loss = total_loss / log_freq
            total_loss = 0.0
            tr.write(f"[step {step:05d}]  train loss {loss:.3f}")

            imgs = jax.device_get(diffusion.eval().sample(16))
            imgs = ((imgs * 0.5 + 0.5) * 255).astype(jnp.uint8)
            imgs = make_image_grid(imgs)
            im = Image.fromarray(imgs)
            im.save(f"sample_{step:05d}.png")


if __name__ == "__main__":
    fire.Fire(train)
