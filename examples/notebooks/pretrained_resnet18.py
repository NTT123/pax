import numpy as np
import pax
import torchvision

IMAGENET_MEAN = np.array((0.485, 0.456, 0.406))
IMAGENET_STD = np.array((0.229, 0.224, 0.225))


def convert_conv(conv, name=None):
    """Return a pax.Conv2D module with weights from pretrained ``conv``."""
    weight = conv.weight.data.contiguous().permute(2, 3, 1, 0).contiguous().numpy()[:]

    pax_conv = pax.Conv2D(
        in_features=conv.in_channels,
        out_features=conv.out_channels,
        kernel_shape=conv.kernel_size,
        stride=conv.stride,
        with_bias=False,
        padding=[(conv.padding[0],) * 2, (conv.padding[1],) * 2],
        data_format="NCHW",
        name=name,
    )
    assert pax_conv.weight.shape == weight.shape
    return pax_conv.replace(weight=weight)


def convert_bn(bn, name=None):
    """Return a pax.BatchNorm2D module from pretrained ``bn``."""
    weight = bn.weight.data.numpy()[None, :, None, None]
    bias = bn.bias.data.numpy()[None, :, None, None]
    running_mean = bn.running_mean.data.numpy()[None, :, None, None]
    running_var = bn.running_var.data.numpy()[None, :, None, None]

    pax_bn = pax.BatchNorm2D(
        num_channels=bias.shape[1],
        create_offset=True,
        create_scale=True,
        decay_rate=0.9,
        eps=1e-5,
        data_format="NCHW",
        name=name,
    )
    assert pax_bn.scale.shape == weight.shape
    assert pax_bn.offset.shape == bias.shape
    assert pax_bn.ema_mean.averages.shape == running_mean.shape
    assert pax_bn.ema_var.averages.shape == running_var.shape

    pax_bn = pax_bn.replace(scale=weight, offset=bias)
    pax_bn = pax_bn.replace_node(pax_bn.ema_mean.averages, running_mean)
    pax_bn = pax_bn.replace_node(pax_bn.ema_var.averages, running_var)
    return pax_bn


def convert_basic_block(block):
    conv1 = convert_conv(block.conv1, name="conv1")
    bn1 = convert_bn(block.bn1, name="bn1")
    conv2 = convert_conv(block.conv2, name="conv2")
    bn2 = convert_bn(block.bn2, name="bn2")

    if block.downsample is not None:
        conv0 = convert_conv(block.downsample[0], name="proj_conv")
        bn0 = convert_bn(block.downsample[1], name="proj_bn")
        return ((conv1, bn1), (conv2, bn2)), (conv0, bn0)
    else:
        return (((conv1, bn1), (conv2, bn2)),)


def convert_block_group(group):
    out = []
    for i in range(len(group)):
        out.append(convert_basic_block(group[i]))
    return out


def convert_linear(linear):
    weight = linear.weight.data.numpy()
    bias = linear.bias.data.numpy()
    pax_linear = pax.Linear(
        in_dim=weight.shape[1], out_dim=weight.shape[0], with_bias=True
    )
    weight = np.transpose(weight)
    assert pax_linear.bias.shape == bias.shape
    assert pax_linear.weight.shape == weight.shape

    return pax_linear.replace(weight=weight, bias=bias)


def load_pretrained_resnet18():
    resnet18 = pax.nets.ResNet18(3, 1000)
    resnet18_pt = torchvision.models.resnet18(pretrained=True).eval()
    pax_resnet = [
        convert_conv(resnet18_pt.conv1),
        convert_bn(resnet18_pt.bn1),
        convert_block_group(resnet18_pt.layer1),
        convert_block_group(resnet18_pt.layer2),
        convert_block_group(resnet18_pt.layer3),
        convert_block_group(resnet18_pt.layer4),
        convert_linear(resnet18_pt.fc),
    ]

    def replace_parts(resnet18):
        # replace resnet18 part by part
        resnet18.initial_conv = pax_resnet[0]
        resnet18.initial_batchnorm = pax_resnet[1]
        for i in range(len(resnet18.block_groups)):
            bg = resnet18.block_groups[i]
            for j in range(len(bg.blocks)):
                b = bg.blocks[j]
                mods = pax_resnet[2 + i][j]
                b.layers = mods[0]
                if b.use_projection:
                    b.proj_conv = mods[1][0]
                    b.proj_batchnorm = mods[1][1]

        resnet18.logits = pax_resnet[-1]
        # make sure we are in `eval` mode when doing evaluation.
        return resnet18.eval()

    return pax.pure(replace_parts)(resnet18)
