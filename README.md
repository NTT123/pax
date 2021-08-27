# Pytree Jax (Pax)

A stateful pytree library for training neural networks.


## Getting started

```python
import pax
import jax


class Counter(pax.Module):
    counter: pax.tree.State

    def __init__(self, start_value: int):
        self.counter = jax.numpy.array(start_value)

    def __call__(self):
        self.counter = self.counter + 1
        return self.counter


print(Counter(3)())
```

## Rules

1. All parameters and states have to be annotated with the type ``pax.tree.Parameter`` or ``pax.tree.State``.
2. Any field whose type is not `pax.tree.State` or  `pax.tree.Parameter` should not be modified inside the forward pass.
3. Functions that are input to jax's higher-order functions (e.g., ``jax.jit``, ``jax.grad``, ``jax.pmap``, etc.) should have no side effects. Modified objects/values should be returned as output of the function.

