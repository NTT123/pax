import jax
import jax.numpy as jnp
import pax


@pax.pure
def test_jit_side_effect():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.counter = 0

        def __call__(self, x):
            self.counter = self.counter + 1
            return x

    m = M()
    assert m.counter == 0
    m(1)
    assert m.counter == 1

    @jax.jit
    def fwd(mod, x):
        return mod(x)

    fwd(m, 1)
    assert m.counter == 1

    @pax.jit_
    def fwd_(mod, x):
        return mod(x)

    assert fwd_(m, 2) == 2
    assert m.counter == 2


@pax.pure
def test_grad_side_effect():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.counter = 0

        def __call__(self, x):
            self.counter = self.counter + 1
            return x

    def fwd(mod, x):
        return mod(x)

    m = M()
    grad_fn = jax.grad(fwd, has_aux=False)
    grad_fn(m, jnp.zeros(()))
    assert m.counter == 0

    grad_fn_ = pax.grad_(fwd, has_aux=False)
    grads = grad_fn_(m, jnp.zeros(()))
    assert isinstance(grads, M)
    assert m.counter == 1

    def fwd_aux(mod, x):
        return mod(x), 1

    grad_fn_ = pax.grad_(fwd_aux, has_aux=True)
    grads, aux = grad_fn_(m, jnp.zeros(()))
    assert isinstance(grads, M)
    assert m.counter == 2
    assert aux == 1


@pax.pure
def test_value_and_grad_side_effect():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.counter = 0

        def __call__(self, x):
            self.counter = self.counter + 1
            return x

    def fwd(mod, x):
        return mod(x)

    m = M()
    vag_fn_ = pax.value_and_grad_(fwd, has_aux=False)
    value, grads = vag_fn_(m, jnp.zeros(()))
    assert isinstance(grads, M)
    assert value == 0
    assert m.counter == 1

    def fwd_aux(mod, x):
        return mod(x), 1

    vag_fn_ = pax.value_and_grad_(fwd_aux, has_aux=True)
    (value, aux), grads = vag_fn_(m, jnp.zeros(()))
    assert m.counter == 2
    assert aux == 1
    assert isinstance(grads, M)
