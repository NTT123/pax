import jax
import pax


def test_run_resnet():
    resnet = pax.nets.ResNet18(3, 1)
    x = jax.numpy.zeros((1, 3, 18, 18))
    y = pax.pure(resnet)(x)
    assert y.shape == (1, 1)


def test_run_transformer():
    transformer = pax.nets.Transformer(8, 2, 2, 0.1)
    x = jax.numpy.zeros((1, 15, 8))
    y = pax.pure(transformer)(x)
    assert y.shape == (1, 15, 8)
