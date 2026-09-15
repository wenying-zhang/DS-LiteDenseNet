"""Network definitions: four DenseNet variants plus two lightweight baselines.

The variants come from one parametrised class and span a 2x2 design,
    {full [6,12,24,16] or lite [4,6,8,6]} x {standard or depthwise-separable},
so parameter and FLOP counts are always derived from the same source.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class LightDepthwiseConv(nn.Module):
    """3x3 depthwise (+BN+ReLU) then 1x1 pointwise (+BN)."""
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, stride, padding, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.net(x)


class _DenseLayer(nn.Module):
    def __init__(self, in_ch, growth, use_ds, use_bottleneck=True):
        super().__init__()
        self.use_bottleneck = use_bottleneck
        bottleneck = growth * 4 if use_bottleneck else in_ch
        self.bn1 = nn.BatchNorm2d(in_ch)
        if use_bottleneck:
            self.conv1 = nn.Conv2d(in_ch, bottleneck, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(bottleneck)
        if use_ds:
            self.main = LightDepthwiseConv(bottleneck, growth)
        else:
            self.main = nn.Conv2d(bottleneck, growth, 3, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        if self.use_bottleneck:
            out = F.relu(self.bn2(self.conv1(out)), inplace=True)
        out = self.main(out)
        return torch.cat([x, out], 1)


class _Transition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x), inplace=True)))


class DenseNetVariant(nn.Module):
    def __init__(self, layers=(6, 12, 24, 16), init_ch=64, growth=32,
                 use_ds=False, use_bottleneck=True, compression=0.5, num_classes=2):
        super().__init__()
        if use_ds:
            self.stem = LightDepthwiseConv(3, init_ch, k=7, stride=2, padding=3)
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, init_ch, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(init_ch), nn.ReLU(inplace=True))
        self.pool0 = nn.MaxPool2d(3, stride=2, padding=1)

        blocks, ch = [], init_ch
        for i, n in enumerate(layers):
            blk = nn.Sequential(*[_DenseLayer(ch + j * growth, growth, use_ds,
                                              use_bottleneck) for j in range(n)])
            blocks.append(blk); ch += n * growth
            if i != len(layers) - 1:
                out = int(ch * compression)
                blocks.append(_Transition(ch, out)); ch = out
        self.features = nn.Sequential(*blocks)
        self.final_bn = nn.BatchNorm2d(ch)          # Grad-CAM target layer
        red = max(ch // 4, 64)
        self.classifier = nn.Sequential(
            nn.Linear(ch, red), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(red, num_classes))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool0(self.stem(x))
        x = self.features(x)
        x = F.relu(self.final_bn(x))
        # x.mean, not F.adaptive_avg_pool2d: the two agree bit for bit in the
        # forward pass, but adaptive_avg_pool2d's CUDA backward accumulates with
        # atomicAdd and has no deterministic implementation, so a network using
        # it is not bit-reproducible even under --deterministic -- which permits
        # such kernels with a warning rather than failing. Reducing with mean
        # removes the only such operation from these four architectures.
        # torchvision's MobileNetV2 still pools this way and is outside our
        # control; ShuffleNetV2 already uses mean.
        x = x.mean(dim=(2, 3))
        return self.classifier(x)


def build_model(name, num_classes=2):
    # Every artefact this repository writes encodes the model name in a filename
    # of the form <tag>_<model>_seed<N>, and operating_point.py and
    # plot_reliability.py recover the model by parsing it back out. An
    # underscore in the name makes that parse ambiguous: the leading part would
    # be absorbed into the tag and the trailing part read as the model, which
    # groups results under the wrong key without raising. Refuse the name here
    # rather than let a downstream mean be silently wrong.
    if "_" in name:
        raise ValueError(
            f"Model name {name!r} contains an underscore. Names are embedded in "
            f"result filenames and parsed back out by delimiter, so an "
            f"underscore would mis-assign that model's results. Use e.g. "
            f"'EfficientNetB0' rather than 'EfficientNet_B0'.")
    n = name.lower()
    if n == "densenet121":
        return DenseNetVariant((6, 12, 24, 16), 64, 32, use_ds=False, num_classes=num_classes)
    if n == "densenet121ds":
        return DenseNetVariant((6, 12, 24, 16), 64, 32, use_ds=True, num_classes=num_classes)
    if n == "litedensenet":
        return DenseNetVariant((4, 6, 8, 6), 32, 32, use_ds=False, num_classes=num_classes)
    if n in ("dslitedensenet", "ours"):
        return DenseNetVariant((4, 6, 8, 6), 32, 32, use_ds=True, num_classes=num_classes)
    if n == "mobilenetv2":
        m = torchvision.models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    if n == "shufflenetv2":
        m = torchvision.models.shufflenet_v2_x1_0(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"unknown model {name}")


def get_target_layer(model, name):
    """Last conv-ish layer with spatial dims, for Grad-CAM."""
    if isinstance(model, DenseNetVariant):
        return model.final_bn
    n = name.lower()
    if n == "mobilenetv2":
        return model.features[-1]
    if n == "shufflenetv2":
        return model.conv5
    raise ValueError(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6  # millions


def count_flops_g(model, device, size=224):
    """GFLOPs via thop if available, else None."""
    try:
        from thop import profile
        x = torch.randn(1, 3, size, size, device=device)
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return 2 * macs / 1e9  # FLOPs ~ 2 * MACs
    except Exception:
        return None
