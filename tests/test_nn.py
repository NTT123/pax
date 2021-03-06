import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pax
import pytest


# def test_batchnorm_train():
#     bn = pax.BatchNorm(
#         3, True, True, 0.9, reduced_axes=[0, 1], param_shape=[1, 1, 3]
#     )
#     bn = pax.enable_train_mode(bn)
#     x = jnp.ones((1, 10, 3))
#     old_state = bn.ema_var.averages
#     y = bn(x)
#     new_state = bn.ema_var.averages
#     chex.assert_tree_all_equal_shapes(old_state, new_state)
#     chex.assert_tree_all_finite(new_state)
#     assert y.shape == (1, 10, 3)


def test_batchnorm1D_train():
    bn = pax.BatchNorm1D(3, True, True, 0.9)
    bn = pax.enable_train_mode(bn)
    x = jnp.ones((1, 10, 3))
    old_state = bn.ema_mean.averages
    bn, y = pax.purecall(bn, x)
    new_state = bn.ema_mean.averages
    chex.assert_tree_all_equal_shapes(old_state, new_state)
    chex.assert_tree_all_finite(new_state)
    assert y.shape == (1, 10, 3)


def test_batchnorm2D_train():
    bn = pax.BatchNorm2D(3, True, True, 0.9)
    bn = pax.enable_train_mode(bn)
    x = jnp.ones((1, 10, 8, 3))
    old_state = bn.scale
    bn, y = pax.purecall(bn, x)
    new_state = bn.scale
    chex.assert_tree_all_equal_shapes(old_state, new_state)
    chex.assert_tree_all_finite(new_state)
    assert y.shape == (1, 10, 8, 3)


def test_batchnorm_no_scale():
    bn = pax.BatchNorm1D(3, False, True, 0.9)
    x = jnp.ones((1, 10, 3))
    bn, _ = pax.purecall(bn, x)


def test_batchnorm_no_offset():
    bn = pax.BatchNorm1D(3, True, False, 0.9)
    x = jnp.ones((1, 10, 3))
    bn, _ = pax.purecall(bn, x)


# def test_batchnorm_eval():
#     bn = pax.BatchNorm(
#         3, True, True, 0.9, reduced_axes=[0, 1], param_shape=[1, 1, 3]
#     )
#     bn = pax.enable_eval_mode(bn)
#     x = jnp.ones((1, 10, 3))
#     old_state = bn.ema_mean
#     y = bn(x)
#     new_state = bn.ema_mean
#     assert y.shape == (1, 10, 3)
#     assert old_state == new_state


# def test_batchnorm_params_filter():
#     bn = pax.BatchNorm(
#         3, True, True, 0.9, reduced_axes=[0, 1], param_shape=[1, 1, 3]
#     )
#     params = pax.select_parameters(bn)
#     bn = bn.update_parameters(params)


def test_conv_1d_basic():
    conv = pax.Conv1D(3, 5, 3, padding="SAME", with_bias=False)
    x = jnp.ones((1, 10, 3), dtype=jnp.float32)
    y = conv(x)
    assert y.shape == (1, 10, 5)


def test_conv_2d_basic():
    conv = pax.Conv2D(3, 5, 3, padding="SAME", with_bias=True)
    x = jnp.ones((1, 10, 10, 3), dtype=jnp.float32)
    y = conv(x)
    assert y.shape == (1, 10, 10, 5)


def test_layer_norm_1():
    """Make sure our LayerNorm behaves the same as hk.LayerNorm."""
    layer_norm = pax.LayerNorm(3, -1, True, True)
    print(layer_norm.summary())
    x = np.random.randn(32, 3).astype(np.float32)
    fwd = hk.transform(lambda x: hk.LayerNorm(-1, True, True)(x))
    rng = jax.random.PRNGKey(42)
    params = fwd.init(rng, x)
    np.testing.assert_array_equal(layer_norm.scale, params["layer_norm"]["scale"])
    np.testing.assert_array_equal(layer_norm.offset, params["layer_norm"]["offset"])
    y1 = fwd.apply(params, None, x)
    y2 = layer_norm(x)
    np.testing.assert_array_equal(y1, y2)


def test_layer_norm_2():
    """Make sure our LayerNorm behaves the same as hk.LayerNorm."""
    layer_norm = pax.LayerNorm(3, -1, False, False)
    print(layer_norm.summary())
    x = np.random.randn(32, 3).astype(np.float32)
    fwd = hk.transform(lambda x: hk.LayerNorm(-1, False, False)(x))
    rng = jax.random.PRNGKey(42)
    params = fwd.init(rng, x)
    # np.testing.assert_array_equal(layer_norm.scale, params["layer_norm"]["scale"])
    # np.testing.assert_array_equal(layer_norm.offset, params["layer_norm"]["offset"])
    y1 = fwd.apply(params, None, x)
    y2 = layer_norm(x)
    np.testing.assert_array_equal(y1, y2)


def test_layer_norm_init():
    """Make sure our LayerNorm behaves the same as hk.LayerNorm."""
    layer_norm = pax.LayerNorm(
        3,
        -1,
        True,
        True,
        scale_init=jax.nn.initializers.normal(),
        offset_init=jax.nn.initializers.normal(),
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
    group_norm = pax.GroupNorm(8, 32, -1)
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
    fc = pax.Linear(1, 1)
    x = jnp.array([[5.0]], dtype=jnp.float32)
    y = fc(x)
    target = x * fc.weight + fc.bias
    assert target.item() == y.item()


def test_linear():
    fc = pax.Linear(5, 7)
    x = jnp.zeros((32, 5), dtype=jnp.float32)
    y = fc(x)
    assert y.shape == (32, 7)
    assert fc.bias is not None


def test_linear_1():
    fc = pax.Linear(5, 7, b_init=jax.nn.initializers.normal())
    rng_key = jax.random.PRNGKey(42)
    x = jax.random.normal(rng_key, (32, 5), dtype=jnp.float32)
    y = fc(x)
    expected_y = jnp.matmul(x, fc.weight) + fc.bias
    np.testing.assert_allclose(expected_y, y)


def test_linear_wo_bias():
    fc = pax.Linear(5, 7, with_bias=False)
    x = jnp.zeros((32, 5), dtype=jnp.float32)
    y = fc(x)
    assert y.shape == (32, 7)
    assert hasattr(fc, "bias") == False


def test_linear_input_shape_error():
    fc = pax.Linear(2, 3, b_init=jax.nn.initializers.normal())
    rng_key = jax.random.PRNGKey(42)
    x = jax.random.normal(rng_key, (2,), dtype=jnp.float32)
    with pytest.raises(AssertionError):
        y = fc(x)


def test_sequential_mix():
    net = pax.Sequential(pax.Linear(1, 2), jax.nn.relu, pax.Linear(2, 3))
    params = net.parameters()
    x = jnp.zeros((2, 1))
    y = net(x)
    assert y.shape == (2, 3)


# def test_sequential_non_mix():
#     net = pax.Sequential(
#         pax.Linear(1, 2),
#         pax.BatchNorm(2, True, True, 0.99, reduced_axes=[0], param_shape=[1, 2]),
#         pax.Linear(2, 3),
#     )
#     params = net.parameters()
#     x = jnp.zeros((2, 1))
#     y = net(x)
#     assert y.shape == (2, 3)


def test_sequential_all_jax():
    net = pax.Sequential(jax.nn.relu, jax.nn.relu, jax.nn.relu)
    params = net.parameters()
    x = jnp.zeros((2, 1))
    y = net(x)
    assert y.shape == (2, 1)


def test_conv_no_bias():
    conv = pax.Conv2D(3, 3, 3, 1, 1, "SAME", False)
    # assert conv.bias == None and "bias" not in conv.pax.name_to_kind


def test_native_conv1d_1():
    rng_key = jax.random.PRNGKey(42)
    conv1d = pax.Conv1D(
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
    conv1d = pax.Conv1D(
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
    conv1d = pax.Conv1D(
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
    conv1d = pax.Conv1D(
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


def test_native_conv1d_5():
    rng_key = jax.random.PRNGKey(42)
    conv1d = pax.Conv1D(
        in_features=8,
        out_features=6,
        kernel_shape=30,
        stride=2,
        rate=3,
        padding="SAME",
        with_bias=False,
        data_format="NWC",
        feature_group_count=2,
        rng_key=rng_key,
    )
    x = jax.random.normal(rng_key, (2, 6, 8))
    y = conv1d(x)

    hk_conv = hk.transform(
        lambda x: hk.Conv1D(
            6, 30, 2, 3, "SAME", False, data_format="NWC", feature_group_count=2
        )(x),
    )
    params = hk_conv.init(rng_key, x)
    hk_y = hk_conv.apply({"conv1_d": {"w": conv1d.weight}}, rng_key, x)
    assert params["conv1_d"]["w"].shape == conv1d.weight.shape
    np.testing.assert_allclose(y, hk_y)


def test_native_conv2d_1():
    rng_key = jax.random.PRNGKey(42)
    conv2d = pax.Conv2D(
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
    conv2d = pax.Conv2D(
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
    conv2d = pax.Conv2D(
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
    conv2d = pax.Conv2D(
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
    conv2d = pax.Conv2D(
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
    conv2d = pax.Conv2D(
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
    conv2d = pax.Conv2D(
        in_features=7,
        out_features=5,
        kernel_shape=(10, 20),
        stride=(1, 1),
        rate=(1, 3),
        padding="VALID",
        with_bias=True,
        data_format="NCHW",
        b_init=jax.nn.initializers.normal(),
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


def test_native_conv2d_8():
    rng_key = jax.random.PRNGKey(46)
    conv2d = pax.Conv2D(
        in_features=9,
        out_features=6,
        kernel_shape=(10, 20),
        stride=(1, 1),
        rate=(1, 3),
        padding="VALID",
        with_bias=True,
        data_format="NCHW",
        feature_group_count=3,
        b_init=jax.nn.initializers.normal(),
        rng_key=rng_key,
    )
    x = jax.random.normal(rng_key, (2, 9, 40, 60))
    y = conv2d(x)

    hk_conv = hk.transform(
        lambda x: hk.Conv2D(
            6,
            (10, 20),
            (1, 1),
            (1, 3),
            "VALID",
            True,
            data_format="NCHW",
            feature_group_count=3,
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
    fc = pax.Linear(3, 5, with_bias=False, rng_key=rng_key)
    hk_fc = hk.transform(lambda x: hk.Linear(5, with_bias=False)(x))
    x = jax.random.normal(rng_key, (7, 4, 3), dtype=jnp.float32)
    y = fc(x)
    hk_y = hk_fc.apply({"linear": {"w": fc.weight}}, rng_key, x)
    np.testing.assert_allclose(y, hk_y)


def test_native_linear_w_bias():
    rng_key = jax.random.PRNGKey(44)
    fc = pax.Linear(
        9,
        5,
        with_bias=True,
        rng_key=rng_key,
        b_init=jax.nn.initializers.normal(),
    )
    hk_fc = hk.transform(lambda x: hk.Linear(5, with_bias=True)(x))
    x = jax.random.normal(rng_key, (7, 11, 9), dtype=jnp.float32)
    y = fc(x)
    hk_y = hk_fc.apply({"linear": {"w": fc.weight, "b": fc.bias}}, rng_key, x)
    np.testing.assert_allclose(y, hk_y)


def test_native_conv2d_transpose_1():
    rng_key = jax.random.PRNGKey(42)
    conv2d_t = pax.Conv2DTranspose(
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
    conv2d_t = pax.Conv2DTranspose(
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
    conv2d_t = pax.Conv2DTranspose(
        in_features=7,
        out_features=5,
        kernel_shape=(8, 8),
        stride=(4, 4),
        padding="SAME",
        with_bias=True,
        data_format="NCHW",
        rng_key=rng_key,
        b_init=jax.nn.initializers.normal(),
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
    conv2d_t = pax.Conv2DTranspose(
        in_features=7,
        out_features=5,
        kernel_shape=(8, 8),
        stride=(4, 4),
        padding="SAME",
        with_bias=True,
        data_format="NHWC",
        rng_key=rng_key,
        b_init=jax.nn.initializers.normal(),
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
    conv1d_t = pax.Conv1DTranspose(
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
    conv1d_t = pax.Conv1DTranspose(
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
    conv1d_t = pax.Conv1DTranspose(
        in_features=7,
        out_features=5,
        kernel_shape=(8,),
        stride=(4,),
        padding="SAME",
        with_bias=True,
        data_format="NCW",
        rng_key=rng_key,
        b_init=jax.nn.initializers.normal(),
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
    conv1d_t = pax.Conv1DTranspose(
        in_features=7,
        out_features=5,
        kernel_shape=(8,),
        stride=(4,),
        padding="SAME",
        with_bias=True,
        data_format="NWC",
        rng_key=rng_key,
        b_init=jax.nn.initializers.normal(),
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
    drop = pax.Dropout(0.9)
    rng_key = jax.random.PRNGKey(42)
    x = jax.random.normal(rng_key, (1, 2, 50), dtype=jnp.float32)
    drop = pax.enable_eval_mode(drop)
    y = drop(x)
    assert y is x
    drop = pax.enable_train_mode(drop)

    drop, y = pax.purecall(drop, x)
    assert jnp.sum(y == 0).item() > 80

    x = jnp.ones_like(x)
    drop, y = pax.purecall(drop, x)
    assert jnp.max(y).item() == 10.0

    with pytest.raises(AssertionError):
        drop = pax.Dropout(1.0)


def test_embed():
    embed = pax.Embed(5, 7)
    x = jnp.array([[[1, 2, 4]]], dtype=jnp.int32)
    y = embed(x)
    assert y.shape == (1, 1, 3, 7)
    hk_embed = hk.transform(lambda x: hk.Embed(5, 7)(x))
    hk_y = hk_embed.apply({"embed": {"embeddings": embed.weight}}, None, x)
    np.testing.assert_allclose(y, hk_y)


def test_vanilla_rnn():
    rnn = pax.VanillaRNN(3, 9)
    state = rnn.initial_state(2)
    assert state.hidden.shape == (2, 9)
    x = jnp.ones((2, 3))
    state, x = rnn(state, x)
    assert state.hidden.shape == (2, 9)
    assert x.shape == (2, 9)


def test_gru():
    gru = pax.GRU(3, 7)
    state = gru.initial_state(2)
    assert state.hidden.shape == (2, 7)
    x = jnp.ones((2, 3), dtype=jnp.float32)
    state, x = gru(state, x)
    assert state.hidden.shape == (2, 7)
    assert x.shape == (2, 7)


def test_lstm():
    lstm = pax.LSTM(3, 7)
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
    y = pax.avg_pool(x, (2, 1), (2, 1), "SAME", -1)
    assert y.shape == (3, 3, 3)


def test_haiku_max_pool():
    x = np.zeros((3, 5, 3), dtype=np.float32)
    y = pax.max_pool(x, (2, 1), (3, 1), "SAME", -1)
    assert y.shape == (3, 2, 3)


def test_conv_wrong_input_size():
    conv1 = pax.Conv2D(3, 6, 3)
    x = jnp.zeros((2, 9, 9, 7), dtype=jnp.float32)
    with pytest.raises(ValueError):
        y = conv1(x)


def test_list_submodules_in_state():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.fc = pax.Linear(2, 2)
            self.state_of_module = pax.Linear(2, 5)

    m = M()
    mods = m.submodules()
    assert len(mods) == 2, "expecting `state_of_module` is counted as a module."


def test_sequential_get_set_item():
    fc1 = pax.Linear(1, 2)
    fc2 = pax.Linear(2, 3)
    fc3 = pax.Linear(2, 1)
    a = pax.Sequential(fc1, jax.nn.relu, fc2)
    assert a[-1] == fc2
    a = a.set(-1, fc3)
    assert a[-1] == fc3
    assert a[0] == fc1


def test_apply_mutate_no_side_effect():
    a = pax.Sequential(pax.Linear(2, 2), pax.Linear(4, 4))

    @pax.pure
    def f(mod):
        mod.test__ = 123
        return mod

    b = a.apply(f)
    assert not hasattr(a[0], "test__")
    assert hasattr(b[0], "test__")


def test_new_method_mutate():
    init_fn = pax.Linear.__init__
    a = pax.Linear(1, 1)
    b = pax.Linear(2, 2)
    assert pax.Linear.__init__ == init_fn


def test_identity_module():
    ident = pax.Identity()
    x = jnp.zeros((3, 3))
    y = ident(x)
    assert jnp.array_equal(x, y) == True


def test_sigmoid():
    @jax.jit
    @jax.vmap
    @jax.grad
    def _sigmoid(x: jnp.ndarray):
        out = 1 / (1 + jnp.exp(-x))
        grad = out * (1 - out)
        t = jax.lax.stop_gradient(grad) * x
        return t + jax.lax.stop_gradient(-t + out)

    x = jnp.linspace(-50, 50, 1000)
    s = jax.jit(jax.vmap(jax.grad(jax.nn.sigmoid)))
    np.testing.assert_array_equal(_sigmoid(x), s(x))


def test_slot():
    class M(pax.Module):
        __slots__ = ()

    with pytest.raises(ValueError):
        m = M()


def test_ema_eval():
    x = jnp.ones((3, 3))
    ema = pax.EMA(x, 0.0, True)
    y = ema.eval()(x)
    np.testing.assert_array_equal(x, y)

    x0 = jnp.zeros((3, 3))
    ema, y = pax.purecall(ema, x0)
    np.testing.assert_array_equal(y, jnp.zeros_like(x))


def test_ema_allow_int_1():
    x = jnp.ones((3, 3), dtype=jnp.int16)
    with pytest.raises(ValueError):
        ema = pax.EMA(x, 0.9)

    ema = pax.EMA(x, 0.9, allow_int=True)


def test_ema_allow_int_2():
    x = (jnp.ones((3, 3), dtype=jnp.int16), jnp.ones((3, 3), dtype=jnp.float16))
    ema = pax.EMA(x, 0.9, allow_int=True)
    ema, x = pax.purecall(ema, x)
    assert x[0].dtype == jnp.int16
    assert x[1].dtype == jnp.float16


def test_ema_allow_int_3():
    x = jnp.ones((3, 3), dtype=jnp.int16) == 0
    assert x.dtype == jnp.bool_
    ema = pax.EMA(x, 0.9, allow_int=True)
    ema, x = pax.purecall(ema, x)
    assert x.dtype == jnp.bool_
