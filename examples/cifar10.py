import jax
import jax.numpy as jnp
import optax
import pax
import tensorflow as tf
import tensorflow_datasets as tfds
from pax.nets import ResNet18
from tqdm.auto import tqdm

batch_size = 128
num_training_steps = 100_000
learning_rate = 1e-4
pax.seed_rng_key(42)


def loss_fn(params: ResNet18, model: ResNet18, inputs) -> pax.utils.LossFnOutput:
    model = model.update(params)
    image, label = inputs["image"], inputs["label"]
    image = image.astype(jnp.float32) / 255.0 * 2 - 1.0

    logits = model(image)
    log_prs = jax.nn.log_softmax(logits, axis=-1)
    log_prs = jax.nn.one_hot(label, num_classes=logits.shape[-1]) * log_prs
    log_prs = jnp.sum(log_prs, axis=-1)
    loss = -jnp.mean(log_prs)
    return loss, (loss, model)


@jax.jit
def test_loss_fn(model, inputs):
    model = model.eval()
    loss = loss_fn(model.parameters(), model, inputs)[0]
    return loss


update_fn = pax.utils.build_update_fn(loss_fn)
fast_update_fn = jax.jit(update_fn)

dataset = tfds.load("cifar10")

dataloader = (
    dataset["train"]
    .cache()
    .repeat()
    .shuffle(50_000)
    .batch(batch_size)
    .take(num_training_steps)
    .prefetch(tf.data.AUTOTUNE)
    .enumerate(1)
)


test_dataloader = (
    dataset["test"]
    .cache()
    .batch(batch_size, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)


net = ResNet18(input_channels=3, num_classes=10)
print(net.summary())
print()
opt = pax.optim.from_optax(optax.adamw(learning_rate))(net.parameters())


total_loss = 0.0
tr = tqdm(dataloader, desc="training")
for step, batch in tr:
    del batch["id"]
    batch = jax.tree_map(lambda x: x.numpy(), batch)
    loss, net, opt = fast_update_fn(net, opt, batch)
    total_loss = total_loss + loss

    if step % 1000 == 0:
        loss = total_loss / 1000.0
        total_loss = 0.0
        tr.write(f"[step {step}]  loss {loss:.3f}")

    if step % 1000 == 0:
        total_test_losses = 0.0
        for test_batch in test_dataloader:
            del test_batch["id"]
            test_batch = jax.tree_map(lambda x: x.numpy(), test_batch)
            loss = test_loss_fn(net, test_batch)
            total_test_losses = loss + total_test_losses
        test_loss = total_test_losses / len(test_dataloader)
        tr.write(f"[step {step}]  test loss {test_loss:.3f}")
