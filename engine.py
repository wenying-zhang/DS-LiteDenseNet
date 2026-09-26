"""Training loop, inference and metric computation.

Model selection uses validation AUROC only; the test split is scored once per
model per seed and never influences training.
"""
import os
import random
import re
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, f1_score, balanced_accuracy_score,
                             roc_curve)

import config as C
from stats import expected_calibration_error


def set_seed(seed, deterministic=False):
    """Seed the generators, and optionally remove run-to-run variation.

    With deterministic=False the run is fast but not bit-reproducible: cuDNN
    picks convolution algorithms by timing them, and several backward kernels
    accumulate in non-deterministic order. Repeating a command then shifts
    AUROC by about as much as the seed-to-seed spread.

    With deterministic=True the same command gives the same numbers on the same
    machine and software stack, at a cost of roughly one third to one half
    again in training time. Two qualifications matter, and both have caught us
    out.

    First, warn_only=True is deliberate: a kernel with no deterministic
    implementation warns and runs anyway rather than aborting the sweep. Until
    the global pooling was changed to a mean reduction in models.py, every
    DenseNet-family architecture here hit exactly that case, so runs labelled
    deterministic were reproducible only to about the width of the seed-to-seed
    spread. torchvision's MobileNetV2 still pools with adaptive_avg_pool2d and
    is not ours to change, so that one model remains in this position.

    Second, the flag makes a *command* repeatable, not a model. A subset run
    (--models, --seeds) repeats the same numbers as another subset run of the
    same subset; whether it also matches the corresponding rows of a larger
    sweep depends on nothing in this function guaranteeing it. Treat a subset
    run as a new execution and keep its output under its own name, which
    run_ablation.py does.

    CUBLAS_WORKSPACE_CONFIG must be set before the CUDA context is created;
    run_ablation.py sets it above `import torch` when --deterministic is in
    argv, so the shell variable below is a fallback rather than a requirement:

        CUBLAS_WORKSPACE_CONFIG=:4096:8 python run_ablation.py --deterministic
    """
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        p = torch.softmax(model(x), dim=1)[:, 1]
        probs.append(p.cpu().numpy()); labels.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(labels)


def youden_threshold(y, prob):
    """Threshold maximising Youden's J, clipped to the probability range.

    sklearn sets thresholds[0] above every observed score (np.inf on current
    versions) so that the curve starts at (0, 0). If no threshold achieves a
    positive J -- a model at or below chance, or a degenerate split -- argmax
    returns that first entry and the function would report a "probability"
    outside [0, 1]. Clipping keeps the returned value interpretable; a model in
    that state is broken either way, and the metrics computed from it will say
    so.
    """
    fpr, tpr, thr = roc_curve(y, prob)
    return float(np.clip(thr[int(np.argmax(tpr - fpr))], 0.0, 1.0))


def compute_metrics(y, prob, thr=0.5):
    y = np.asarray(y).astype(int)
    pred = (prob >= thr).astype(int)
    out = {"thr": float(thr)}
    out["auroc"] = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
    out["auprc"] = average_precision_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
    out["acc"] = float((pred == y).mean())
    out["bacc"] = float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out["sens"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    out["spec"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    out["f1"] = float(f1_score(y, pred, zero_division=0))
    out["ece"] = expected_calibration_error(y, prob)
    return out


def train_model(model, train_loader, val_loader, device,
                epochs=C.EPOCHS, lr=C.LR, weight_decay=C.WEIGHT_DECAY,
                patience=C.EARLY_PATIENCE, verbose=True):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    crit = nn.CrossEntropyLoss()

    best_auc, best_state, wait = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
        sched.step()

        vprob, vy = predict(model, val_loader, device)
        vauc = roc_auc_score(vy, vprob) if len(np.unique(vy)) > 1 else 0.0
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  val_auroc={vauc:.4f}")
        if vauc > best_auc:
            best_auc, wait = vauc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  early stop @ epoch {ep} (best val_auroc={best_auc:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


_TV_DENSENET_KEY = re.compile(
    r"^features\.(?:"
    r"(?P<stem>conv0|norm0)"
    r"|denseblock(?P<blk>\d+)\.denselayer(?P<lyr>\d+)\.(?P<lmod>norm1|conv1|norm2|conv2)"
    r"|transition(?P<trn>\d+)\.(?P<tmod>norm|conv)"
    r"|(?P<final>norm5)"
    r")\.(?P<param>.+)$")


def _torchvision_densenet_to_variant(key):
    """Map a torchvision DenseNet state-dict key onto DenseNetVariant's naming.

    models.DenseNetVariant with (6, 12, 24, 16), 64 initial channels and growth
    32 has exactly the encoder of torchvision's densenet121 -- the same layers,
    shapes and order -- but names them differently:

        torchvision                                   DenseNetVariant
        features.conv0 / norm0                        stem.0 / stem.1
        features.denseblock<b>.denselayer<l>.norm1    features.<2(b-1)>.<l-1>.bn1
                                        .conv1 / norm2 / conv2   .conv1 / .bn2 / .main
        features.transition<t>.norm / conv            features.<2t-1>.bn / .conv
        features.norm5                                final_bn

    Only the classifier differs, and it is excluded from transfer anyway.
    Returns None for keys with no counterpart (the classifier).
    """
    m = _TV_DENSENET_KEY.match(key)
    if m is None:
        return None
    p = m.group("param")
    if m.group("stem"):
        return f"stem.{0 if m.group('stem') == 'conv0' else 1}.{p}"
    if m.group("blk"):
        b, l = int(m.group("blk")), int(m.group("lyr"))
        sub = {"norm1": "bn1", "conv1": "conv1", "norm2": "bn2", "conv2": "main"}[m.group("lmod")]
        return f"features.{2 * (b - 1)}.{l - 1}.{sub}.{p}"
    if m.group("trn"):
        t = int(m.group("trn"))
        return f"features.{2 * t - 1}.{'bn' if m.group('tmod') == 'norm' else 'conv'}.{p}"
    return f"final_bn.{p}"


def load_imagenet_encoder(model, model_name, device):
    """Initialise a torchvision baseline from ImageNet weights, head excluded.

    Only DenseNet121, MobileNetV2 and ShuffleNetV2 have published ImageNet
    weights; LiteDenseNet, DenseNet121DS and DS-LiteDenseNet are architectures
    defined in this repository and have none. A sweep run with --imagenet is
    therefore not a like-for-like comparison, and the caller is told so per
    model rather than left to infer it from the absence of a message. The head
    is dropped on the same rule as load_pretrained_encoder, so only the encoder
    transfers.

    DenseNet121 is built by models.DenseNetVariant, not by torchvision, so its
    parameter names differ from the published checkpoint's and a name-for-name
    match loads nothing (an earlier version of this function did exactly
    that, printing "loaded 0 ImageNet tensors" and training from random
    initialisation under the _imagenet tag). The keys are now translated by
    _torchvision_densenet_to_variant. MobileNetV2 and ShuffleNetV2 are
    torchvision's own classes and match by name.

    Every encoder tensor of the published checkpoint must land. A partial or
    empty transfer raises rather than prints, because a run that silently
    trains from scratch under a pretrained tag is indistinguishable from a
    genuine one in every file it writes.
    """
    import torchvision.models as tvm
    builders = {
        "densenet121": lambda: tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1),
        "mobilenetv2": lambda: tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1),
        "shufflenetv2": lambda: tvm.shufflenet_v2_x1_0(
            weights=tvm.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
    }
    key = model_name.lower()
    if key not in builders:
        print(f"  {model_name}: no ImageNet weights exist for this architecture; "
              f"trained from scratch. This run is an unequal comparison by design.")
        return
    sd = builders[key]().state_dict()
    _HEADS = ("classifier", "fc")
    encoder = {k: v for k, v in sd.items()
               if not any(h in k.split(".") for h in _HEADS)}
    if key == "densenet121":
        encoder = {_torchvision_densenet_to_variant(k): v for k, v in encoder.items()}
        if None in encoder:
            raise RuntimeError("DenseNet121: an ImageNet encoder key has no "
                               "counterpart in DenseNetVariant.")
    msd = model.state_dict()
    missing = [k for k in encoder if k not in msd]
    shape_bad = [k for k in encoder if k in msd and encoder[k].shape != msd[k].shape]
    if missing or shape_bad:
        raise RuntimeError(
            f"{model_name}: {len(missing)} ImageNet encoder tensors have no "
            f"counterpart in the model and {len(shape_bad)} differ in shape "
            f"(e.g. {(missing + shape_bad)[:3]}). Refusing to train a partially "
            f"initialised network under the _imagenet tag.")
    msd.update(encoder)
    model.load_state_dict(msd)
    model.to(device)
    n_head = len(sd) - len(encoder)
    print(f"  {model_name}: loaded {len(encoder)} ImageNet tensors, all of the "
          f"encoder ({n_head} head tensors left at random initialisation)")


def load_pretrained_encoder(model, ckpt_path, device):
    """Load self-supervised pretrained weights, skipping the classifier head."""
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    msd = model.state_dict()
    # Exclude every head this repository builds. DenseNetVariant and
    # MobileNetV2 name theirs "classifier"; torchvision's ShuffleNetV2 names
    # its head "fc", and without the second test a pretrained ShuffleNetV2
    # checkpoint of the same class count would carry a fitted classifier into
    # a run reported as encoder-only pretraining.
    _HEADS = ("classifier", "fc")
    keep = {k: v for k, v in sd.items()
            if k in msd and not any(h in k.split(".") for h in _HEADS)
            and v.shape == msd[k].shape}
    if not keep:
        # Same failure as the ImageNet loader had: a checkpoint whose names do
        # not match loads nothing, and the run would train from scratch under
        # the _ssl tag without saying so.
        raise RuntimeError(f"No tensor in {ckpt_path} matches this model by name "
                           f"and shape; it was saved from a different architecture "
                           f"or with a different naming scheme.")
    msd.update(keep); model.load_state_dict(msd)
    print(f"  loaded {len(keep)} pretrained tensors from {ckpt_path}")
    return model
