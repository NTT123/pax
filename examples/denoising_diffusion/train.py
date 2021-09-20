import jax
import opax
import pax
import tensorflow as tf

from data_loader import load_celeb_a
from model import GaussianDiffusion, UNet


def train(
    batch_size: int = 32,
    num_training_steps: int = 100_000,
):

    model = UNet(dim=64, dim_mults=(1, 2, 4, 8))

    diffusion = GaussianDiffusion(
        model,
        image_size=64,
        timesteps=1000,
        loss_type="l1",  # L1 or L2
    )

    diffusion.parameters()

    print(diffusion.summary())

    training_images = jax.random.normal(jax.random.PRNGKey(42), (2, 64, 64, 3))
    loss = diffusion(training_images)
    print(loss)

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

    optimizer = opax.adamw(1e-4)(diffusion.parameters())

    from tqdm.auto import tqdm

    for step, batch in enumerate(tqdm(dataloader)):
        batch = jax.tree_map(lambda x: x.numpy(), batch)
        loss, diffusion, optimizer = fast_update_fn(diffusion, optimizer, batch)


train()

if __name__ == "__main__":
    import fire

    fire.Fire(train)
