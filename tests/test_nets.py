import jax
import pax


def test_run_resnet():
    resnet = pax.nets.ResNet18(3, 1)
    x = jax.numpy.zeros((1, 3, 18, 18))

    @pax.pure
    def _run(resnet):
        return resnet(x)

    y = _run(resnet)
    assert y.shape == (1, 1)


def test_run_transformer():
    transformer = pax.nets.Transformer(8, 2, 2, 0.1)
    x = jax.numpy.zeros((1, 15, 8))

    @pax.pure
    def _run(transformer):
        return transformer(x)

    y = _run(transformer)
    assert y.shape == (1, 15, 8)
