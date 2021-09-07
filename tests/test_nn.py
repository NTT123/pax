import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pax


def test_batchnorm_train():
    bn = pax.nn.BatchNorm((None, None, 3), True, True, 0.9)
    bn = bn.train()
    x = jnp.ones((1, 10, 3))
    old_state = bn.state
    y = bn(x)
    new_state = bn.state
    chex.assert_tree_all_equal_shapes(old_state, new_state)
    chex.assert_tree_all_finite(new_state)
    assert y.shape == (1, 10, 3)


def test_batchnorm1D_train():
    bn = pax.nn.BatchNorm1D(3, True, True, 0.9)
    bn = bn.train()
    x = jnp.ones((1, 10, 3))
    old_state = bn.state
    y = bn(x)
    new_state = bn.state
    chex.assert_tree_all_equal_shapes(old_state, new_state)
    chex.assert_tree_all_finite(new_state)
    assert y.shape == (1, 10, 3)


def test_batchnorm2D_train():
    bn = pax.nn.BatchNorm2D(3, True, True, 0.9)
    bn = bn.train()
    x = jnp.ones((1, 10, 8, 3))
    old_state = bn.state
    y = bn(x)
    new_state = bn.state
    chex.assert_tree_all_equal_shapes(old_state, new_state)
    chex.assert_tree_all_finite(new_state)
    assert y.shape == (1, 10, 8, 3)


def test_batchnorm_eval():
    bn = pax.nn.BatchNorm((None, None, 3), True, True, 0.9)
    bn = bn.eval()
    x = jnp.ones((1, 10, 3))
    old_state = bn.state
    y = bn(x)
    new_state = bn.state
    assert y.shape == (1, 10, 3)
    assert old_state == new_state


def test_batchnorm_params_filter():
    bn = pax.nn.BatchNorm((None, None, 3), True, True, 0.9)
    params = bn.filter("parameter")
    bn = bn.update(params)


def test_conv_1d_basic():
    conv = pax.nn.Conv1D(3, 5, 3, padding="SAME", with_bias=False)
    x = jnp.ones((1, 10, 3), dtype=jnp.float32)
    y = conv(x)
    assert y.shape == (1, 10, 5)


def test_conv_2d_basic():
    conv = pax.nn.Conv2D(3, 5, 3, padding="SAME", with_bias=True)
    x = jnp.ones((1, 10, 10, 3), dtype=jnp.float32)
    y = conv(x)
    assert y.shape == (1, 10, 10, 5)


def test_layer_norm_1():
    """Make sure our LayerNorm behaves the same as hk.LayerNorm."""
    layer_norm = pax.nn.LayerNorm(3, -1, True, True)
    x = np.empty((32, 3), dtype=np.float32)
    fwd = hk.transform(lambda x: hk.LayerNorm(-1, True, True)(x))
    rng = jax.random.PRNGKey(42)
    params = fwd.init(rng, x)
    np.testing.assert_array_equal(layer_norm.scale, params["layer_norm"]["scale"])
    np.testing.assert_array_equal(layer_norm.offset, params["layer_norm"]["offset"])


def test_layer_norm_init():
    """Make sure our LayerNorm behaves the same as hk.LayerNorm."""
    layer_norm = pax.nn.LayerNorm(
        3,
        -1,
        True,
        True,
        scale_init=pax.initializers.random_normal(),
        offset_init=pax.initializers.truncated_normal(),
    )
    x = np.empty((32, 3), dtype=np.float32)
    fwd = hk.transform(
        lambda x: hk.LayerNorm(
            -1,
            True,
            True,
            scale_init=hk.initializers.RandomNormal(),
            offset_init=hk.initializers.TruncatedNormal(),
        )(x)
    )
    rng = jax.random.PRNGKey(42)
    params = fwd.init(rng, x)
    chex.assert_equal_shape((layer_norm.scale, params["layer_norm"]["scale"]))
    chex.assert_equal_shape((layer_norm.offset, params["layer_norm"]["offset"]))


def test_linear_computation():
    fc = pax.nn.Linear(1, 1)
    x = jnp.array([[5.0]], dtype=jnp.float32)
    y = fc(x)
    target = x * fc.weight + fc.bias
    assert target.item() == y.item()


def test_linear():
    fc = pax.nn.Linear(5, 7)
    x = jnp.zeros((32, 5), dtype=jnp.float32)
    y = fc(x)
    assert y.shape == (32, 7)
    assert fc.bias is not None


def test_linear_wo_bias():
    fc = pax.nn.Linear(5, 7, with_bias=False)
    x = jnp.zeros((32, 5), dtype=jnp.float32)
    y = fc(x)
    assert y.shape == (32, 7)

    assert fc.bias is None


def test_sequential_mix():
    net = pax.nn.Sequential(pax.nn.Linear(1, 2), jax.nn.relu, pax.nn.Linear(2, 3))
    params = net.parameters()
    x = jnp.zeros((2, 1))
    y = net(x)
    assert y.shape == (2, 3)


def test_sequential_non_mix():
    net = pax.nn.Sequential(
        pax.nn.Linear(1, 2),
        pax.nn.BatchNorm([None, 2], True, True, 0.99),
        pax.nn.Linear(2, 3),
    )
    params = net.parameters()
    x = jnp.zeros((2, 1))
    y = net(x)
    assert y.shape == (2, 3)


def test_sequential_all_jax():
    net = pax.nn.Sequential(jax.nn.relu, jax.nn.relu, jax.nn.relu)
    params = net.parameters()
    x = jnp.zeros((2, 1))
    y = net(x)
    assert y.shape == (2, 1)
