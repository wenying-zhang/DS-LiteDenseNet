"""Measure what a deployed model actually costs: latency and activation memory.

Table 2 reports parameters and FLOPs. Neither is what an offline device is
limited by. Latency is set by memory traffic as much as by arithmetic, and
DenseNet-style concatenation moves a lot of memory per FLOP, so a model with the
smallest parameter count can still be the slowest. Peak activation memory, not
the weight file, is usually what decides whether a network fits at all.

This script measures both, on CPU by default because that is the deployment
target the manuscript describes. It needs no data, no checkpoints and no
training: what a forward pass costs depends on the architecture and the input
size, not on what the weights contain.

The activation columns follow from the architecture alone and reproduce exactly.
The timings do not. Eighteen consecutive runs of this script on the same idle
24-thread processor gave DenseNet-121 medians from 39.4 to 56.3 ms, and every
model varied by 22 to 37 per cent of its own median, so a single run can misstate
a model by a third. The ratios between models are far steadier: over those runs
the depthwise-separable variant was slower than its dense counterpart in every
one, and the model ordering was stable in sixteen.

Run this several times and take the minimum, which is what Table 2 of the
manuscript reports: contention only ever adds time, so the shortest run is the
closest estimate of what the network itself costs. Read the ratios rather than
the absolute milliseconds, and record the thread count with any figure taken
from here.

Usage:
    python bench_inference.py                        # CPU, batch 1, all models
    python bench_inference.py --device cuda
    python bench_inference.py --threads 4            # pin to a core budget
    python bench_inference.py --batch 8

Writes results/inference_cost.csv and prints a table. These figures complement
the parameter and FLOP counts rather than replacing them; they answer a
different question.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

import config as C
from models import build_model, count_params

MODELS = ["DenseNet121", "DenseNet121DS", "LiteDenseNet",
          "DSLiteDenseNet", "MobileNetV2", "ShuffleNetV2"]


def peak_activation_bytes(model, x):
    """Two activation figures, neither of which is the true working set.

    largest_tensor is the biggest single intermediate. It is a floor on what
    must fit at once, and in practice it is set by the early high-resolution
    layers, so it says little about what a densely connected block costs later.

    total is the sum over every leaf output: not memory, since tensors are
    freed as the pass proceeds, but a proxy for how much activation data the
    forward pass writes and reads. It is the figure that tracks latency on a
    bandwidth-bound device.

    An exact working set needs a per-runtime allocator trace; treat both as
    architectural indicators rather than as a device measurement.
    """
    peak = [0]
    total = [0]
    handles = []

    def hook(_m, _inp, out):
        outs = out if isinstance(out, (tuple, list)) else [out]
        for o in outs:
            if torch.is_tensor(o):
                b = o.numel() * o.element_size()
                peak[0] = max(peak[0], b)
                total[0] += b

    for m in model.modules():
        if len(list(m.children())) == 0:          # leaf modules only
            handles.append(m.register_forward_hook(hook))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return peak[0], total[0]


def time_model(model, x, device, warmup, runs):
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return np.asarray(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--batch", type=int, default=1,
                    help="1 is the point-of-care case: one radiograph at a time")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads; pin this so two runs are at least "
                         "comparable, and report the value alongside the timing")
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    x = torch.randn(args.batch, 3, C.IMG_SIZE, C.IMG_SIZE, device=device)

    print(f"device={args.device}  threads={torch.get_num_threads()}  "
          f"batch={args.batch}  input={C.IMG_SIZE}x{C.IMG_SIZE}  "
          f"runs={args.runs} after {args.warmup} warmup\n")

    rows = []
    for name in args.models:
        model = build_model(name).to(device).eval()
        t = time_model(model, x, device, args.warmup, args.runs)
        act, act_total = peak_activation_bytes(model, x)
        params = count_params(model)
        rows.append({
            "model": name,
            "params_M": params,
            "ms_median": float(np.median(t)),
            "ms_p05": float(np.percentile(t, 5)),
            "ms_p95": float(np.percentile(t, 95)),
            "ms_per_image": float(np.median(t)) / args.batch,
            "largest_tensor_MB": act / 1e6,
            "total_activation_MB": act_total / 1e6,
            "weights_MB": params * 1e6 * 4 / 1e6,
        })
        del model

    df = pd.DataFrame(rows)
    # One invocation is one sample of a noisy quantity, and it overwrites the
    # last. Say so where the number is produced, not only in the documentation.
    print("\n[note] this is ONE run. Repeated runs of this command on an idle "
          "machine have varied by 22 to 37 per cent per model; Table 2 of the "
          "manuscript is the minimum over eighteen. Compare ratios between "
          "models rather than absolute milliseconds, and re-run before quoting "
          "a timing.")
    ref = df.loc[df.model == "DSLiteDenseNet", "ms_median"]
    if len(ref):
        df["ms_vs_ours"] = df.ms_median / float(ref.iloc[0])

    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    out = C.RESULTS_DIR / "inference_cost.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print("\nA parameter count bounds the weight file, not the latency and not "
          "the activation working set. Comparing each variant with its "
          "depthwise-separable counterpart isolates the cost of the "
          "factorisation itself.")


if __name__ == "__main__":
    main()
