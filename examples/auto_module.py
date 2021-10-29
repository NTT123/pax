import jax
import jax.numpy as jnp
import pax

pax.seed_rng_key(42)


class M(pax.AutoModule):
    def __call__(self, x):
        x = self.get_or_create("fc1", lambda: pax.nn.Linear(3, 3))(x)
        x = jax.nn.relu(x)
        x = self.get_or_create("fc2", lambda: pax.nn.Linear(3, 5))(x)
        return x

    def inference(self, x):
        x = self.fc1(x)
        x = jax.nn.relu(x)
        x = self.fc2(x)
        return x


def loss_fn(net: M, x, y):
    net, y_hat = net % x
    loss = jnp.mean(jnp.square(y - y_hat))
    return loss, net


x = jnp.ones((32, 3))
y = jnp.ones((32, 5))
net, _ = M() % x

print(net.summary())
y = net.inference(x)
y = net(x)
print(y.shape)
