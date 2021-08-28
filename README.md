# Pax

Pax is a stateful pytree library for training neural networks.


## Getting started

```python
import pax
import jax


class Counter(pax.Module):
    counter: pax.State

    def __init__(self, start_value: int = 0):
        self.counter = jax.numpy.array(start_value)

    def __call__(self):
        self.counter = self.counter + 1
        return self.counter


print(Counter(3)())
```

## Rules and limitations

1. All parameters and states have to be annotated with the type ``pax.Parameter`` or ``pax.State``.
2. All sub-modules have to be annotated with the type `pax.Module`.
3. PyTree-compatible types such as: `List[pax.Parameter]`, `Dict[str, pax.Module]`, `Optional[pax.State]` are allowed.
4. Tuple types such as `Tuple[pax.Parameter, pax.Parameter]` are not allowed. Use `List` instead.
4. Any field whose type is not `pax.State` or  `pax.Parameter` should not be modified inside the forward pass.
5. Functions (e.g., loss functions) that are input to jax's higher-order functions (e.g., ``jax.jit``, ``jax.grad``, ``jax.pmap``, etc.) should have no side effects. Modified objects/values should be returned as output of the function.
