import jax
import jax.numpy as jnp
import optax
import pax
import tensorflow_datasets as tfds
from pax.nets import ResNet18
from tqdm.auto import tqdm

batch_size = 32
num_training_steps = 10_000
learning_rate = 1e-4

def loss_fn(params: ResNet18, model: ResNet18, inputs) -> pax.utils.LossFnOutput:
    model = model.update(params)
    image, label = inputs["image"], inputs["label"]
    image = image.astype(jnp.float32) / 255.0 * 2 - 1.0

    logits = model(image)
    log_prs = jax.nn.log_softmax(logits, axis=-1)
    log_prs = jax.nn.one_hot(label, num_classes=logits.shape[-1]) * logits
    log_prs = jnp.sum(log_prs, axis=-1)

    loss = -jnp.mean(log_prs)
    return loss, (loss, model)


update_fn = pax.utils.build_update_fn(loss_fn)
fast_update_fn = jax.jit(update_fn)

dataset = tfds.load("cifar10")

dataloader = (
    dataset["train"]
    .cache()
    .shuffle(50_000)
    .batch(batch_size)
    .take(num_training_steps)
    .prefetch(1)
    .enumerate(1)
    .as_numpy_iterator()
)

net = ResNet18(input_channels=3, num_classes=10)
print(net.summary())
opt = pax.optim.from_optax(optax.adamw(learning_rate))(net.parameters())

total_loss = 0.0
for step, batch in tqdm(dataloader, decs='training'):
    del batch['id']
    loss, net, opt = fast_update_fn(net, opt, batch)
    total_loss = total_loss + loss
    if step % 100 == 0:
        loss = total_loss / 100.0
        total_loss = 0.0
        print(f"[step {step}]  loss {loss:.3f}")
