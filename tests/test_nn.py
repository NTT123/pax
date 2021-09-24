import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pax
import pytest


def test_batchnorm_train():
    bn = pax.nn.batch_norm.BatchNorm(
        (None, None, 3), True, True, 0.9, reduced_axes=[0, 1], param_shape=[1, 1, 3]
    )
    bn = pax.enable_train_mode(bn)
    x = jnp.ones((1, 10, 3))
    old_state = bn.ema_var.averages
    y = bn(x)
    new_state = bn.ema_var.averages
    chex.assert_tree_all_equal_shapes(old_state, new_state)
    chex.assert_tree_all_finite(new_state)
    assert y.shape == (1, 10, 3)


def test_batchnorm1D_train():
    bn = pax.nn.BatchNorm1D(3, True, True, 0.9)
    bn = pax.enable_train_mode(bn)
    x = jnp.ones((1, 10, 3))
    old_state = bn.ema_mean.averages
    y = bn(x)
    new_state = bn.ema_mean.averages
    chex.assert_tree_all_equal_shapes(old_state, new_state)
    chex.assert_tree_all_finite(new_state)
    assert y.shape == (1, 10, 3)


def test_batchnorm2D_train():
    bn = pax.nn.BatchNorm2D(3, True, True, 0.9)
    bn = pax.enable_train_mode(bn)
    x = jnp.ones((1, 10, 8, 3))
    old_state = bn.scale
    y = bn(x)
    new_state = bn.scale
    chex.assert_tree_all_equal_shapes(old_state, new_state)
    chex.assert_tree_all_finite(new_state)
    assert y.shape == (1, 10, 8, 3)


def test_batchnorm_eval():
    bn = pax.nn.batch_norm.BatchNorm(
        3, True, True, 0.9, reduced_axes=[0, 1], param_shape=[1, 1, 3]
    )
    bn = pax.enable_eval_mode(bn)
    x = jnp.ones((1, 10, 3))
    old_state = bn.ema_mean
    y = bn(x)
    new_state = bn.ema_mean
    assert y.shape == (1, 10, 3)
    assert old_state == new_state


def test_batchnorm_params_filter():
    bn = pax.nn.batch_norm.BatchNorm(
        3, True, True, 0.9, reduced_axes=[0, 1], param_shape=[1, 1, 3]
    )
    params = pax.select_parameter(bn)
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


def test_group_norm_1():
    """Make sure our GroupNorm behaves the same as hk.GroupNorm."""
    group_norm = pax.nn.GroupNorm(8, 32, -1)
    x = np.random.randn(32, 4, 4, 32).astype(np.float32)
    fwd = hk.transform(lambda x: hk.GroupNorm(8, -1, True, True)(x))
    rng = jax.random.PRNGKey(42)
    params = fwd.init(rng, x)
    np.testing.assert_array_equal(group_norm.scale, params["group_norm"]["scale"])
    np.testing.assert_array_equal(group_norm.offset, params["group_norm"]["offset"])
    o1 = group_norm(x)
    o2 = fwd.apply(params, rng, x)
    np.testing.assert_array_equal(o1, o2)


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


def test_linear_1():
    fc = pax.nn.Linear(5, 7, b_init=pax.initializers.random_normal())
    rng_key = jax.random.PRNGKey(42)
    x = jax.random.normal(rng_key, (32, 5), dtype=jnp.float32)
    y = fc(x)
    expected_y = jnp.matmul(x, fc.weight) + fc.bias
    np.testing.assert_allclose(expected_y, y)


def test_linear_wo_bias():
    fc = pax.nn.Linear(5, 7, with_bias=False)
    x = jnp.zeros((32, 5), dtype=jnp.float32)
    y = fc(x)
    assert y.shape == (32, 7)
    assert hasattr(fc, "bias") == False


def test_linear_input_shape_error():
    fc = pax.nn.Linear(2, 3, b_init=pax.initializers.random_normal())
    rng_key = jax.random.PRNGKey(42)
    x = jax.random.normal(rng_key, (2,), dtype=jnp.float32)
    with pytest.raises(AssertionError):
        y = fc(x)


def test_sequential_mix():
    net = pax.nn.Sequential(pax.nn.Linear(1, 2), jax.nn.relu, pax.nn.Linear(2, 3))
    params = pax.select_parameter(net)
    x = jnp.zeros((2, 1))
    y = net(x)
    assert y.shape == (2, 3)


def test_sequential_non_mix():
    net = pax.nn.Sequential(
        pax.nn.Linear(1, 2),
        pax.nn.batch_norm.BatchNorm(
            [None, 2], True, True, 0.99, reduced_axes=[0], param_shape=[1, 2]
        ),
        pax.nn.Linear(2, 3),
    )
    params = pax.select_parameter(net)
    x = jnp.zeros((2, 1))
    y = net(x)
    assert y.shape == (2, 3)


def test_sequential_all_jax():
    net = pax.nn.Sequential(jax.nn.relu, jax.nn.relu, jax.nn.relu)
    params = pax.select_parameter(net)
    x = jnp.zeros((2, 1))
    y = net(x)
    assert y.shape == (2, 1)


def test_native_conv1d_1():
    rng_key = jax.random.PRNGKey(42)
    conv1d = pax.nn.Conv1D(
        in_features=3,
        out_features=5,
        kernel_shape=3,
        stride=1,
        rate=1,
        padding=[(1, 1)],
        with_bias=True,
        data_format="NWC",
        rng_key=rng_key,
    )
    x = jax.random.normal(rng_key, (2, 6, 3))
    y = conv1d(x)
    assert y.shape == (2, 6, 5)


def test_native_conv1d_2():
    rng_key = jax.random.PRNGKey(42)
    conv1d = pax.nn.Conv1D(
        in_features=7,
        out_features=5,
        kernel_shape=3,
        stride=1,
        rate=1,
        padding=[(1, 1)],
        with_bias=False,
        data_format="NWC",
        rng_key=rng_key,
    )
    x = jax.random.normal(rng_key, (2, 6, 7))
    y = conv1d(x)
    assert y.shape == (2, 6, 5)

    hk_conv = hk.transform(
        lambda x: hk.Conv1D(5, 3, 1, 1, [(1, 1)], False, data_format="NWC")(x),
    )
    params = hk_conv.init(rng_key, x)
    hk_y = hk_conv.apply({"conv1_d": {"w": conv1d.weight}}, rng_key, x)
    np.testing.assert_allclose(y, hk_y)


def test_native_conv1d_3():
    rng_key = jax.random.PRNGKey(42)
    conv1d = pax.nn.Conv1D(
        in_features=7,
        out_features=5,
        kernel_shape=3,
        stride=2,
        rate=1,
        padding=[(1, 1)],
        with_bias=False,
        data_format="NWC",
        rng_key=rng_key,
    )
    x = jax.random.normal(rng_key, (2, 6, 7))
    y = conv1d(x)

    hk_conv = hk.transform(
        lambda x: hk.Conv1D(5, 3, 2, 1, [(1, 1)], False, data_format="NWC")(x),
    )
    params = hk_conv.init(rng_key, x)
    hk_y = hk_conv.apply({"conv1_d": {"w": conv1d.weight}}, rng_key, x)
    assert params["conv1_d"]["w"].shape == conv1d.weight.shape
    np.testing.assert_allclose(y, hk_y)


def test_native_conv1d_4():
    rng_key = jax.random.PRNGKey(42)
    conv1d = pax.nn.Conv1D(
        in_features=7,
        out_features=5,
        kernel_shape=30,
        stride=2,
        rate=3,
        padding="SAME",
        with_bias=False,
        data_format="NWC",
        rng_key=rng_key,
    )
    x = jax.random.normal(rng_key, (2, 6, 7))
    y = conv1d(x)

    hk_conv = hk.transform(
        lambda x: hk.Conv1D(5, 30, 2, 3, "SAME", False, data_format="NWC")(x),
    )
    params = hk_conv.init(rng_key, x)
    hk_y = hk_conv.apply({"conv1_d": {"w": conv1d.weight}}, rng_key, x)
    assert params["conv1_d"]["w"].shape == conv1d.weight.shape
    np.testing.assert_allclose(y, hk_y)


def test_native_conv2d_1():
    rng_key = jax.random.PRNGKey(42)
    conv2d = pax.nn.Conv2D(
        in_features=3,
        out_features=5,
        kernel_shape=3,
        stride=1,
        rate=1,
        padding=[(1, 1), (1, 1)],
        with_bias=True,
        data_format="NHWC",
        rng_key=rng_key,
    )
    x = jax.random.normal(rng_key, (2, 6, 7, 3))
    y = conv2d(x)
    assert y.shape == (2, 6, 7, 5)


def test_native_conv2d_2():
    rng_key = jax.random.PRNGKey(11)
    conv2d = pax.nn.Conv2D(
        in_features=7,
        out_features=5,
        kernel_shape=(20, 30),
        stride=(2, 3),
        rate=(2, 3),
        padding="SAME",
        with_bias=False,
        data_format="NHWC",
        rng_key=rng_key,
    )
    x = jax.random.normal(rng_key, (2, 40, 60, 7))
    y = conv2d(x)

    hk_conv = hk.transform(
        lambda x: hk.Conv2D(
            5, (20, 30), (2, 3), (2, 3), "SAME", False, data_format="NHWC"
        )(x),
    )
    params = hk_conv.init(rng_key, x)
    hk_y = hk_conv.apply({"conv2_d": {"w": conv2d.weight}}, rng_key, x)
    assert params["conv2_d"]["w"].shape == conv2d.weight.shape
    np.testing.assert_allclose(y, hk_y)


def test_native_conv2d_3():
    rng_key = jax.random.PRNGKey(55)
    conv2d = pax.nn.Conv2D(
        in_features=7,
        out_features=5,
        kernel_shape=(20, 30),
        stride=(2, 3),
        rate=(2, 3),
        padding="VALID",
        with_bias=False,
        data_format="NHWC",
        rng_key=rng_key,
    )
    x = jax.random.normal(rng_key, (2, 40, 60, 7))
    y = conv2d(x)

    hk_conv = hk.transform(
        lambda x: hk.Conv2D(
            5, (20, 30), (2, 3), (2, 3), "VALID", False, data_format="NHWC"
        )(x),
    )
    params = hk_conv.init(rng_key, x)
    hk_y = hk_conv.apply({"conv2_d": {"w": conv2d.weight}}, rng_key, x)
    assert params["conv2_d"]["w"].shape == conv2d.weight.shape
    np.testing.assert_allclose(y, hk_y)


def test_native_conv2d_4():
    rng_key = jax.random.PRNGKey(66)
    conv2d = pax.nn.Conv2D(
        in_features=7,
        out_features=5,
        kernel_shape=(20, 30),
        stride=(1, 1),
        rate=(2, 3),
        padding="VALID",
        with_bias=False,
        data_format="NHWC",
        rng_key=rng_key,
    )
    x = jax.random.normal(rng_key, (2, 40, 60, 7))
    y = conv2d(x)

    hk_conv = hk.transform(
        lambda x: hk.Conv2D(
            5, (20, 30), (1, 1), (2, 3), "VALID", False, data_format="NHWC"
        )(x),
    )
    params = hk_conv.init(rng_key, x)
    hk_y = hk_conv.apply({"conv2_d": {"w": conv2d.weight}}, rng_key, x)
    assert params["conv2_d"]["w"].shape == conv2d.weight.shape
    np.testing.assert_allclose(y, hk_y)


def test_native_conv2d_5():
    rng_key = jax.random.PRNGKey(99)
    conv2d = pax.nn.Conv2D(
        in_features=7,
        out_features=5,
        kernel_shape=(10, 20),
        stride=(1, 1),
        rate=(1, 3),
        padding="VALID",
        with_bias=False,
        data_format="NHWC",
        rng_key=rng_key,
    )
    x = jax.random.normal(rng_key, (2, 40, 60, 7))
    y = conv2d(x)

    hk_conv = hk.transform(
        lambda x: hk.Conv2D(
            5, (10, 20), (1, 1), (1, 3), "VALID", False, data_format="NHWC"
        )(x),
    )
    params = hk_conv.init(rng_key, x)
    hk_y = hk_conv.apply({"conv2_d": {"w": conv2d.weight}}, rng_key, x)
    assert params["conv2_d"]["w"].shape == conv2d.weight.shape
    np.testing.assert_allclose(y, hk_y)


def test_native_conv2d_6():
    rng_key = jax.random.PRNGKey(40)
    conv2d = pax.nn.Conv2D(
        in_features=7,
        out_features=5,
        kernel_shape=(10, 20),
        stride=(1, 1),
        rate=(1, 3),
        padding="VALID",
        with_bias=False,
        data_format="NCHW",
        rng_key=rng_key,
    )
    x = jax.random.normal(rng_key, (2, 7, 40, 60))
    y = conv2d(x)

    hk_conv = hk.transform(
        lambda x: hk.Conv2D(
            5, (10, 20), (1, 1), (1, 3), "VALID", False, data_format="NCHW"
        )(x),
    )
    params = hk_conv.init(rng_key, x)
    hk_y = hk_conv.apply({"conv2_d": {"w": conv2d.weight}}, rng_key, x)
    assert params["conv2_d"]["w"].shape == conv2d.weight.shape
    np.testing.assert_allclose(y, hk_y)


def test_native_conv2d_7():
    rng_key = jax.random.PRNGKey(46)
    conv2d = pax.nn.Conv2D(
        in_features=7,
        out_features=5,
        kernel_shape=(10, 20),
        stride=(1, 1),
        rate=(1, 3),
        padding="VALID",
        with_bias=True,
        data_format="NCHW",
        b_init=pax.initializers.truncated_normal(),
        rng_key=rng_key,
    )
    x = jax.random.normal(rng_key, (2, 7, 40, 60))
    y = conv2d(x)

    hk_conv = hk.transform(
        lambda x: hk.Conv2D(
            5, (10, 20), (1, 1), (1, 3), "VALID", True, data_format="NCHW"
        )(x),
    )
    params = hk_conv.init(rng_key, x)
    hk_y = hk_conv.apply(
        {"conv2_d": {"w": conv2d.weight, "b": conv2d.bias[:, None, None]}}, rng_key, x
    )
    assert params["conv2_d"]["w"].shape == conv2d.weight.shape
    np.testing.assert_allclose(y, hk_y)


def test_native_linear_wo_bias():
    rng_key = jax.random.PRNGKey(44)
    fc = pax.nn.Linear(3, 5, with_bias=False, rng_key=rng_key)
    hk_fc = hk.transform(lambda x: hk.Linear(5, with_bias=False)(x))
    x = jax.random.normal(rng_key, (7, 4, 3), dtype=jnp.float32)
    y = fc(x)
    hk_y = hk_fc.apply({"linear": {"w": fc.weight}}, rng_key, x)
    np.testing.assert_allclose(y, hk_y)


def test_native_linear_w_bias():
    rng_key = jax.random.PRNGKey(44)
    fc = pax.nn.Linear(
        9,
        5,
        with_bias=True,
        rng_key=rng_key,
        b_init=pax.initializers.truncated_normal(),
    )
    hk_fc = hk.transform(lambda x: hk.Linear(5, with_bias=True)(x))
    x = jax.random.normal(rng_key, (7, 11, 9), dtype=jnp.float32)
    y = fc(x)
    hk_y = hk_fc.apply({"linear": {"w": fc.weight, "b": fc.bias}}, rng_key, x)
    np.testing.assert_allclose(y, hk_y)


def test_native_conv2d_transpose_1():
    rng_key = jax.random.PRNGKey(42)
    conv2d_t = pax.nn.Conv2DTranspose(
        in_features=3,
        out_features=5,
        kernel_shape=3,
        stride=1,
        with_bias=False,
        data_format="NHWC",
        rng_key=rng_key,
    )
    x = jax.random.normal(rng_key, (2, 6, 7, 3))
    y = conv2d_t(x)
    assert y.shape == (2, 6, 7, 5)


def test_native_conv2d_transpose_2():
    rng_key = jax.random.PRNGKey(45)
    conv2d_t = pax.nn.Conv2DTranspose(
        in_features=7,
        out_features=5,
        kernel_shape=(10, 20),
        stride=(1, 1),
        padding="VALID",
        with_bias=False,
        data_format="NHWC",
        rng_key=rng_key,
    )
    x = jax.random.normal(rng_key, (2, 40, 60, 7))
    y = conv2d_t(x)

    hk_conv = hk.transform(
        lambda x: hk.Conv2DTranspose(
            5,
            kernel_shape=(10, 20),
            stride=(1, 1),
            padding="VALID",
            with_bias=False,
            data_format="NHWC",
        )(x),
    )
    params = hk_conv.init(rng_key, x)
    hk_y = hk_conv.apply({"conv2_d_transpose": {"w": conv2d_t.weight}}, rng_key, x)
    assert params["conv2_d_transpose"]["w"].shape == conv2d_t.weight.shape
    np.testing.assert_allclose(y, hk_y)


def test_native_conv2d_transpose_3():
    rng_key = jax.random.PRNGKey(45)
    conv2d_t = pax.nn.Conv2DTranspose(
        in_features=7,
        out_features=5,
        kernel_shape=(8, 8),
        stride=(4, 4),
        padding="SAME",
        with_bias=True,
        data_format="NCHW",
        rng_key=rng_key,
        b_init=pax.initializers.truncated_normal(),
    )
    x = jax.random.normal(rng_key, (2, 7, 40, 60))
    y = conv2d_t(x)

    hk_conv = hk.transform(
        lambda x: hk.Conv2DTranspose(
            5,
            kernel_shape=(8, 8),
            stride=(4, 4),
            padding="SAME",
            with_bias=True,
            data_format="NCHW",
        )(x),
    )
    params = hk_conv.init(rng_key, x)
    hk_y = hk_conv.apply(
        {
            "conv2_d_transpose": {
                "w": conv2d_t.weight,
                "b": conv2d_t.bias[:, None, None],
            }
        },
        rng_key,
        x,
    )
    assert params["conv2_d_transpose"]["w"].shape == conv2d_t.weight.shape
    np.testing.assert_allclose(y, hk_y)


def test_native_conv2d_transpose_4():
    rng_key = jax.random.PRNGKey(45)
    conv2d_t = pax.nn.Conv2DTranspose(
        in_features=7,
        out_features=5,
        kernel_shape=(8, 8),
        stride=(4, 4),
        padding="SAME",
        with_bias=True,
        data_format="NHWC",
        rng_key=rng_key,
        b_init=pax.initializers.truncated_normal(),
    )
    x = jax.random.normal(rng_key, (2, 40, 60, 7))
    y = conv2d_t(x)

    hk_conv = hk.transform(
        lambda x: hk.Conv2DTranspose(
            5,
            kernel_shape=(8, 8),
            stride=(4, 4),
            padding="SAME",
            with_bias=True,
            data_format="NHWC",
        )(x),
    )
    params = hk_conv.init(rng_key, x)
    hk_y = hk_conv.apply(
        {"conv2_d_transpose": {"w": conv2d_t.weight, "b": conv2d_t.bias}}, rng_key, x
    )
    assert params["conv2_d_transpose"]["w"].shape == conv2d_t.weight.shape
    np.testing.assert_allclose(y, hk_y)


def test_native_conv1d_transpose_1():
    rng_key = jax.random.PRNGKey(42)
    conv1d_t = pax.nn.Conv1DTranspose(
        in_features=3,
        out_features=5,
        kernel_shape=3,
        stride=1,
        with_bias=False,
        data_format="NWC",
        rng_key=rng_key,
    )
    x = jax.random.normal(rng_key, (2, 6, 3))
    y = conv1d_t(x)
    assert y.shape == (2, 6, 5)


def test_native_conv1d_transpose_2():
    rng_key = jax.random.PRNGKey(45)
    conv1d_t = pax.nn.Conv1DTranspose(
        in_features=7,
        out_features=5,
        kernel_shape=(10,),
        stride=(1,),
        padding="VALID",
        with_bias=False,
        data_format="NWC",
        rng_key=rng_key,
    )
    x = jax.random.normal(rng_key, (2, 40, 7))
    y = conv1d_t(x)

    hk_conv = hk.transform(
        lambda x: hk.Conv1DTranspose(
            5,
            kernel_shape=(10,),
            stride=(1,),
            padding="VALID",
            with_bias=False,
            data_format="NWC",
        )(x),
    )
    params = hk_conv.init(rng_key, x)
    hk_y = hk_conv.apply({"conv1_d_transpose": {"w": conv1d_t.weight}}, rng_key, x)
    assert params["conv1_d_transpose"]["w"].shape == conv1d_t.weight.shape
    np.testing.assert_allclose(y, hk_y)


def test_native_conv1d_transpose_3():
    rng_key = jax.random.PRNGKey(45)
    conv1d_t = pax.nn.Conv1DTranspose(
        in_features=7,
        out_features=5,
        kernel_shape=(8,),
        stride=(4,),
        padding="SAME",
        with_bias=True,
        data_format="NCW",
        rng_key=rng_key,
        b_init=pax.initializers.truncated_normal(),
    )
    x = jax.random.normal(rng_key, (2, 7, 40))
    y = conv1d_t(x)

    hk_conv = hk.transform(
        lambda x: hk.Conv1DTranspose(
            5,
            kernel_shape=(8,),
            stride=(4,),
            padding="SAME",
            with_bias=True,
            data_format="NCW",
        )(x),
    )
    params = hk_conv.init(rng_key, x)
    hk_y = hk_conv.apply(
        {
            "conv1_d_transpose": {
                "w": conv1d_t.weight,
                "b": conv1d_t.bias[:, None],
            }
        },
        rng_key,
        x,
    )
    assert params["conv1_d_transpose"]["w"].shape == conv1d_t.weight.shape
    np.testing.assert_allclose(y, hk_y)


def test_native_conv1d_transpose_4():
    rng_key = jax.random.PRNGKey(45)
    conv1d_t = pax.nn.Conv1DTranspose(
        in_features=7,
        out_features=5,
        kernel_shape=(8,),
        stride=(4,),
        padding="SAME",
        with_bias=True,
        data_format="NWC",
        rng_key=rng_key,
        b_init=pax.initializers.truncated_normal(),
    )
    x = jax.random.normal(rng_key, (2, 40, 7))
    y = conv1d_t(x)

    hk_conv = hk.transform(
        lambda x: hk.Conv1DTranspose(
            5,
            kernel_shape=(8,),
            stride=(4,),
            padding="SAME",
            with_bias=True,
            data_format="NWC",
        )(x),
    )
    params = hk_conv.init(rng_key, x)
    hk_y = hk_conv.apply(
        {"conv1_d_transpose": {"w": conv1d_t.weight, "b": conv1d_t.bias}}, rng_key, x
    )
    assert params["conv1_d_transpose"]["w"].shape == conv1d_t.weight.shape
    np.testing.assert_allclose(y, hk_y)


def test_dropout():
    drop = pax.nn.Dropout(0.9)
    rng_key = jax.random.PRNGKey(42)
    x = jax.random.normal(rng_key, (1, 2, 50), dtype=jnp.float32)
    drop = pax.enable_eval_mode(drop)
    y = drop(x)
    assert y is x
    drop = pax.enable_train_mode(drop)
    y = drop(x)
    assert jnp.sum(y == 0).item() > 80

    x = jnp.ones_like(x)
    y = drop(x)
    assert jnp.max(y).item() == 10.0

    with pytest.raises(AssertionError):
        drop = pax.nn.Dropout(1.0)


def test_embed():
    embed = pax.nn.Embed(5, 7)
    x = jnp.array([[[1, 2, 4]]], dtype=jnp.int32)
    y = embed(x)
    assert y.shape == (1, 1, 3, 7)
    hk_embed = hk.transform(lambda x: hk.Embed(5, 7)(x))
    hk_y = hk_embed.apply({"embed": {"embeddings": embed.weight}}, None, x)
    np.testing.assert_allclose(y, hk_y)


def test_gru():
    gru = pax.nn.GRU(3, 7)
    state = gru.initial_state(2)
    assert state.hidden.shape == (2, 7)
    x = jnp.ones((2, 3), dtype=jnp.float32)
    state, x = gru(state, x)
    assert state.hidden.shape == (2, 7)
    assert x.shape == (2, 7)


def test_lstm():
    lstm = pax.nn.LSTM(3, 7)
    state = lstm.initial_state(2)
    assert state.hidden.shape == (2, 7)
    assert state.cell.shape == (2, 7)
    x = jnp.ones((2, 3), dtype=jnp.float32)
    state, x = lstm(state, x)
    assert state.hidden.shape == (2, 7)
    assert state.cell.shape == (2, 7)
    assert x.shape == (2, 7)


def test_avg_pool():
    x = np.zeros((3, 5, 3), dtype=np.float32)
    y = pax.nn.avg_pool(x, (2, 1), (2, 1), "SAME", -1)
    assert y.shape == (3, 3, 3)


def test_haiku_max_pool():
    x = np.zeros((3, 5, 3), dtype=np.float32)
    y = pax.nn.max_pool(x, (2, 1), (3, 1), "SAME", -1)
    assert y.shape == (3, 2, 3)


def test_conv_wrong_input_size():
    conv1 = pax.nn.Conv2D(3, 6, 3)
    x = jnp.zeros((2, 9, 9, 7), dtype=jnp.float32)
    with pytest.raises(ValueError):
        y = conv1(x)


def test_list_sub_modules_in_state():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.fc = pax.nn.Linear(2, 2)
            self.register_state_subtree("state_of_module", pax.nn.Linear(2, 5))

    m = M()
    mods = m.sub_modules()
    assert len(mods) == 1, "expecting `state_of_module` is not counted as a module"


def test_sequential_get_set_item():
    fc1 = pax.nn.Linear(1, 2)
    fc2 = pax.nn.Linear(2, 3)
    fc3 = pax.nn.Linear(2, 1)
    a = pax.nn.Sequential(fc1, jax.nn.relu, fc2)
    assert a[-1] == fc2
    a[-1] = fc3
    assert a[-1] == fc3
    assert a[0] == fc1


def test_apply_no_side_effect():
    a = pax.nn.Sequential(pax.nn.Linear(2, 2), pax.nn.Linear(4, 4))

    def f(mod):
        with pax.ctx.mutable():
            mod.test__ = 123
        return mod

    b = a.apply(f)
    assert not hasattr(a[0], "test__")
    assert hasattr(b[0], "test__")


def test_new_method_no_side_effects():
    init_fn = pax.nn.Linear.__init__
    a = pax.nn.Linear(1, 1)
    b = pax.nn.Linear(2, 2)
    assert pax.nn.Linear.__init__ == init_fn


def test_identity_module():
    ident = pax.nn.Identity()
    x = jnp.zeros((3, 3))
    y = ident(x)
    assert jnp.array_equal(x, y) == True
