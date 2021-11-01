# Follow resnet implementation at:
# https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/nets/resnet.py
# which is under Apache License, Version 2.0.

"""Resnet Modules."""

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp

from ..core import Module
from ..nn import BatchNorm2D, Conv2D, Linear, max_pool


class ResnetBlock(Module):
    """ResnetBlock"""

    layers: Sequence[Tuple[Conv2D, BatchNorm2D]]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride,
        use_projection: bool,
        bottleneck: bool,
    ):
        super().__init__()
        self.use_projection = use_projection

        if self.use_projection:
            self.proj_conv = Conv2D(
                in_channels,
                out_channels,
                kernel_shape=1,
                stride=stride,
                with_bias=False,
                padding=[(0, 0), (0, 0)],
                data_format="NCHW",
                name="proj_conv",
            )
            self.proj_batchnorm = BatchNorm2D(
                out_channels, True, True, 0.9, data_format="NCHW", name="proj_bn"
            )

        channel_div = 4 if bottleneck else 1
        conv_0 = Conv2D(
            in_features=in_channels,
            out_features=out_channels // channel_div,
            kernel_shape=1 if bottleneck else 3,
            stride=1 if bottleneck else stride,
            with_bias=False,
            padding=[(0, 0), (0, 0)] if bottleneck else [(1, 1), (1, 1)],
            data_format="NCHW",
            name="conv1",
        )

        bn_0 = BatchNorm2D(
            out_channels // channel_div, True, True, 0.9, data_format="NCHW", name="bn1"
        )

        conv_1 = Conv2D(
            in_features=out_channels // channel_div,
            out_features=out_channels,
            kernel_shape=3,
            stride=stride if bottleneck else 1,
            with_bias=False,
            padding=[(1, 1), (1, 1)],
            data_format="NCHW",
            name="conv2",
        )

        bn_1 = BatchNorm2D(
            out_channels, True, True, 0.9, data_format="NCHW", name="bn2"
        )

        layers = ((conv_0, bn_0), (conv_1, bn_1))

        if bottleneck:
            conv_2 = Conv2D(
                in_features=out_channels,
                out_features=out_channels,
                kernel_shape=1,
                stride=1,
                with_bias=False,
                padding=[(0, 0), (0, 0)],
                data_format="NCHW",
                name="conv3",
            )
            bn_2 = BatchNorm2D(
                out_channels,
                True,
                True,
                0.9,
                data_format="NCHW",
                name="bn3",
            )
            layers = layers + ((conv_2, bn_2),)

        self.layers = layers

    def __call__(self, inputs):
        out = shortcut = inputs

        if self.use_projection:
            shortcut = self.proj_conv(shortcut)
            shortcut = self.proj_batchnorm(shortcut)

        for i, (conv_i, bn_i) in enumerate(self.layers):
            out = conv_i(out)
            out = bn_i(out)

            if i < len(self.layers) - 1:
                out = jax.nn.relu(out)

        return jax.nn.relu(out + shortcut)


class BlockGroup(Module):
    """Group of Blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride,
        bottleneck: bool,
        use_projection,
    ):
        super().__init__()

        blocks = []

        for i in range(num_blocks):
            blocks.append(
                ResnetBlock(
                    in_channels=(in_channels if i == 0 else out_channels),
                    out_channels=out_channels,
                    stride=(1 if i else stride),
                    use_projection=(i == 0 and use_projection),
                    bottleneck=bottleneck,
                )
            )

        self.blocks = blocks

    def __call__(self, inputs):
        out = inputs
        for block in self.blocks:
            out = block(out)

        return out


def check_length(length, value, name):
    if len(value) != length:
        raise ValueError(f"`{name}` must be of length 4 not {len(value)}")


class ResNet(Module):
    """A generic ResNet module."""

    CONFIGS = {
        18: {
            "blocks_per_group": (2, 2, 2, 2),
            "bottleneck": False,
            "channels_per_group": (64, 128, 256, 512),
            "use_projection": (False, True, True, True),
        },
        34: {
            "blocks_per_group": (3, 4, 6, 3),
            "bottleneck": False,
            "channels_per_group": (64, 128, 256, 512),
            "use_projection": (False, True, True, True),
        },
        50: {
            "blocks_per_group": (3, 4, 6, 3),
            "bottleneck": True,
            "channels_per_group": (256, 512, 1024, 2048),
            "use_projection": (True, True, True, True),
        },
        101: {
            "blocks_per_group": (3, 4, 23, 3),
            "bottleneck": True,
            "channels_per_group": (256, 512, 1024, 2048),
            "use_projection": (True, True, True, True),
        },
        152: {
            "blocks_per_group": (3, 8, 36, 3),
            "bottleneck": True,
            "channels_per_group": (256, 512, 1024, 2048),
            "use_projection": (True, True, True, True),
        },
        200: {
            "blocks_per_group": (3, 24, 36, 3),
            "bottleneck": True,
            "channels_per_group": (256, 512, 1024, 2048),
            "use_projection": (True, True, True, True),
        },
    }

    def __init__(
        self,
        input_channels: int,
        blocks_per_group: Sequence[int],
        num_classes: int,
        bottleneck: bool = True,
        channels_per_group: Sequence[int] = (256, 512, 1024, 2048),
        use_projection: Sequence[bool] = (True, True, True, True),
        logits_config=None,
        initial_conv_config=None,
        name=None,
    ):
        super().__init__(name=name)

        check_length(4, blocks_per_group, "blocks_per_group")
        check_length(4, channels_per_group, "channels_per_group")

        logits_config = dict(logits_config or {})
        logits_config.setdefault("w_init", jax.nn.initializers.zeros)

        initial_conv_config = dict(initial_conv_config or {})
        initial_conv_config.setdefault("in_features", input_channels)
        initial_conv_config.setdefault("out_features", 64)
        initial_conv_config.setdefault("kernel_shape", 7)
        initial_conv_config.setdefault("stride", 2)
        initial_conv_config.setdefault("with_bias", False)
        initial_conv_config.setdefault("padding", [(3, 3), (3, 3)])
        initial_conv_config.setdefault("data_format", "NCHW")

        self.initial_conv = Conv2D(**initial_conv_config, name="conv1")

        self.initial_batchnorm = BatchNorm2D(
            initial_conv_config["out_features"],
            True,
            True,
            0.9,
            data_format="NCHW",
            name="bn1",
        )

        block_groups = []
        strides = (1, 2, 2, 2)
        for i in range(4):
            block_groups.append(
                BlockGroup(
                    in_channels=(
                        initial_conv_config["out_features"]
                        if i == 0
                        else channels_per_group[i - 1]
                    ),
                    out_channels=channels_per_group[i],
                    num_blocks=blocks_per_group[i],
                    stride=strides[i],
                    bottleneck=bottleneck,
                    use_projection=use_projection[i],
                )
            )

        self.block_groups = block_groups

        self.logits = Linear(
            channels_per_group[-1], num_classes, **logits_config, name="fc"
        )

    def __call__(self, inputs):
        out = inputs
        out = self.initial_conv(out)
        out = self.initial_batchnorm(out)
        out = jax.nn.relu(out)
        out = jnp.pad(out, [(0, 0), (0, 0), (1, 1), (1, 1)])
        out = max_pool(
            out,
            window_shape=(1, 1, 3, 3),
            strides=(1, 1, 2, 2),
            padding="VALID",
            channel_axis=1,
        )
        for block_group in self.block_groups:
            out = block_group(out)

        out = jnp.mean(out, axis=(2, 3))
        return self.logits(out)


class ResNet18(ResNet):
    """ResNet18."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        logits_config=None,
        initial_conv_config=None,
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            initial_conv_config=initial_conv_config,
            logits_config=logits_config,
            **ResNet.CONFIGS[18],
            name="ResNet18",
        )


class ResNet34(ResNet):
    """ResNet34."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        logits_config=None,
        initial_conv_config=None,
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            initial_conv_config=initial_conv_config,
            logits_config=logits_config,
            **ResNet.CONFIGS[34],
        )


class ResNet50(ResNet):
    """ResNet50."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        logits_config=None,
        initial_conv_config=None,
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            initial_conv_config=initial_conv_config,
            logits_config=logits_config,
            **ResNet.CONFIGS[50],
        )


class ResNet101(ResNet):
    """ResNet101."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        logits_config=None,
        initial_conv_config=None,
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            initial_conv_config=initial_conv_config,
            logits_config=logits_config,
            **ResNet.CONFIGS[101],
        )


class ResNet152(ResNet):
    """ResNet152."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        logits_config=None,
        initial_conv_config=None,
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            initial_conv_config=initial_conv_config,
            logits_config=logits_config,
            **ResNet.CONFIGS[152],
        )


class ResNet200(ResNet):
    """ResNet200."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        logits_config=None,
        initial_conv_config=None,
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            initial_conv_config=initial_conv_config,
            logits_config=logits_config,
            **ResNet.CONFIGS[200],
        )
