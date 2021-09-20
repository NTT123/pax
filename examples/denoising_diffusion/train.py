import jax
import opax
import pax
import tensorflow as tf
from PIL import Image

from data_loader import load_celeb_a
from model import GaussianDiffusion, UNet


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
        dataset.cache()
        .repeat()
        .shuffle(len(dataset))
        .batch(batch_size)
        .take(num_training_steps)
        .prefetch(tf.data.AUTOTUNE)
    )

    def loss_fn(params, model, inputs) -> pax.LossFnOutput:
        model = model.update(params)
        loss = model(inputs)
        return loss, (loss, model)

    update_fn = pax.utils.build_update_fn(loss_fn)
    fast_update_fn = pax.jit(update_fn)

    optimizer = opax.adam(learning_rate)(diffusion.parameters())

    from tqdm.auto import tqdm

    total_loss = 0.0
    tr = tqdm(dataloader)
    for step, batch in enumerate(tr, 1):
        batch = jax.tree_map(lambda x: x.numpy(), batch)
        loss, diffusion, optimizer = fast_update_fn(diffusion, optimizer, batch)
        total_loss = total_loss + loss

        if step % log_freq == 0:
            loss = total_loss / log_freq
            total_loss = 0.0
            tr.write(f"[step {step:05d}]  train loss {loss:.3f}")

            img = jax.device_get(diffusion.sample(1)[0])
            img = ((img * 0.5 + 0.5) * 255).astype(jax.numpy.uint8)
            im = Image.fromarray(img)
            im.save(f"sample_{step:05d}.png")


if __name__ == "__main__":
    import fire
    fire.Fire(train)
