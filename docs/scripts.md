# Script reference

This document describes every script in the repository: what it computes, how to
call it, what it writes, and which result in the manuscript or its Supplementary
Material depends on it. The README gives the end-to-end order in which to run
them; this file is the reference for each one.

All scripts read paths and shared settings from `config.py` and write to
`results/` (tables, CSVs, per-seed prediction files), `checkpoints/` (weights) and
`figures/` (plots). None of these directories is tracked. Run every command from
the repository root.

Manuscript numbering used below follows the submitted manuscript: main-text
Sections 3.1–3.6 (methods) and 4.1–4.6 (results), Tables 1–7, Figures 1–7, and
Supplementary Sections S1–S22 with Tables S1–S22. A map of the Supplementary
numbering is at the end.

---

## Contents

| Group | Scripts |
|---|---|
| Shared modules | `config.py`, `data_manifests.py`, `datasets.py`, `models.py`, `engine.py`, `stats.py`, `lung_seg.py` |
| Training | `run_ablation.py`, `run_cross_source.py` |
| Scoring existing checkpoints | `eval_external.py`, `border_occlusion.py` |
| Analyses of saved outputs | `paired_external.py`, `paired_bootstrap.py`, `operating_point.py`, `gradcam_stats.py`, `geometric_reference.py` |
| Localisation | `run_gradcam.py` |
| Calibration plots | `plot_reliability.py`, `export_reliability_curve.py` |
| Diagnostics and cost | `diagnose_tb_domain.py`, `bench_inference.py` |
| Figures | `paper_figures/*.py` |

---

## Shared modules

### `config.py`

Paths, hyper-parameters and preprocessing switches shared by every script.
Normally only `DATA_ROOT` needs changing: it defaults to a `data/` directory next
to the code, and `RAW_DATA` lists each corpus relative to it (RSNA, CheXpert,
VinDr-CXR, Shenzhen, Montgomery with its manual masks, COVIDGR-1.0). The same
file fixes the input size (224), the batch size (32), the epoch budget, the five
seeds, the list of ablation models, and the output directories `manifests/`,
`results/`, `checkpoints/` and `figures/`, which it creates on import.

### `data_manifests.py`

```bash
python data_manifests.py
```

Builds the unified manifests from the raw corpora and writes them to
`manifests/`:

| File | Content |
|---|---|
| `pneumonia_internal.csv` | RSNA; Lung Opacity positive, Normal negative, the intermediate class discarded |
| `pneumonia_external.csv` | CheXpert (frontal, Pneumonia vs No Finding) and VinDr, `split == external` |
| `tb_internal.csv` | Shenzhen |
| `tb_external.csv` | Montgomery and VinDr |
| `covid_internal.csv` | COVIDGR-1.0; single source, no external arm |

Every row carries `image_id, path, file_type, label, patient_id, source, split`.
Labels are binary. The internal split is patient-wise and stratified by label
(70/15/15), so no patient appears in more than one partition; in these corpora
each patient contributes one frontal radiograph. VinDr labels are the majority
of three radiologist reads: an image is kept when at least two readers marked the
target pathology (positive) or "No finding" (negative), which discards about 26%
of the images. Manifests hold absolute paths and are not tracked; rebuild them on
each machine.

### `datasets.py`

The dataset class used by every training and scoring script, driven by the
manifests. Images are decoded with format-aware handling (DICOM VOI LUT and
`MONOCHROME1` inversion; percentile windowing of other high-bit-depth images),
equalised with CLAHE (clip limit 2.0, 8×8 tiles), resized to 224×224,
standardised per image to zero mean and unit variance, and replicated to three
channels. No image is cropped to the lung field. Training images receive
horizontal flips, rotations up to 10° and brightness and contrast jitter up to
0.2. The module also provides the class-balanced sampler and the worker seeding
used with `--deterministic`.

### `models.py`

The four DenseNet variants come from one parametrised class spanning a 2×2
design, {full `[6,12,24,16]` or lite `[4,6,8,6]`} × {standard or
depthwise-separable}: `DenseNet121`, `DenseNet121DS`, `LiteDenseNet` and
`DSLiteDenseNet`. All four share the same two-layer head. `MobileNetV2` and
`ShuffleNetV2` wrap the torchvision definitions with their own classifiers.
`count_flops_g` gives the FLOP counts of Table 2. Global pooling in the four
DenseNet variants reduces with `mean`, which has a deterministic CUDA backward.

### `engine.py`

Training loop (AdamW, cosine annealing, early stopping on validation AUROC),
inference, and metric computation. Model selection uses validation AUROC only;
the locked test split is scored once per model and seed and never influences
training. The operating threshold is Youden's *J* on the validation split,
carried unchanged to the test split.

### `stats.py`

| Function | Returns | Used for |
|---|---|---|
| `delong_roc_test(y, prob_a, prob_b)` | `(auc_a, auc_b, p)` | single-seed comparison of correlated ROC curves |
| `bootstrap_auc_ci(y, prob)` | `(mean, lo, hi)` | 95% bootstrap interval, 2000 resamples |
| `paired_ttest(a, b)` | `(t, p)` | two-sided paired test across seeds |
| `expected_calibration_error(y, prob)` | ECE | 15 equal-width confidence bins |
| `noninferiority_test(ref, ours, margin)` | `(mean_d, bound, p, ni)` | upper one-sided 95% bound on the gap, `t(0.95, n−1)` |

### `lung_seg.py`

Lung-field masks from the ChestX-Det PSPNet distributed by TorchXRayVision, taken
as the union of the left and right lung channels; Montgomery's manual masks are
used where they exist. The mask is used for measurement only (Grad-CAM lung
energy, the area references, and the lung-aware occlusion regions); nothing in
the repository crops an image to it. The first call downloads about 260 MB of
weights.

---

## Training

### `run_ablation.py` — architecture ablation on a locked test split

```bash
python run_ablation.py --task pneumonia --deterministic      # Table 3
python run_ablation.py --task tb --deterministic             # Table 4
python run_ablation.py --task covid --deterministic          # Table S16
python run_ablation.py --task pneumonia --label_fraction 0.05   # Table 7
python run_ablation.py --task tb --imagenet                  # Table S8
```

For each model and seed: train on the training split, select the best epoch by
validation AUROC, score the locked test split once. Reports mean and standard
deviation over seeds, bootstrap intervals, the paired *t*-test and the
non-inferiority bound against DS-LiteDenseNet, and a DeLong test on one seed fixed
in advance.

| Argument | Default | Meaning |
|---|---|---|
| `--task` | `pneumonia` | `pneumonia`, `tb` or `covid` |
| `--label_fraction` | `1.0` | fraction of training labels used; validation and test fixed |
| `--imagenet` | off | initialise the three torchvision backbones from ImageNet; the other three train from scratch in the same sweep |
| `--pretrained` | none | load an encoder checkpoint (self-supervised initialisation) |
| `--models`, `--seeds` | all six, `config.SEEDS` | restrict the run |
| `--epochs` | `config.EPOCHS` | override the epoch budget, e.g. `--epochs 2` to check the plumbing |
| `--ni_margin` | `0.02` | non-inferiority margin δ |
| `--from_npz` | off | rebuild the aggregate files from the saved predictions, training nothing |
| `--deterministic` | off | deterministic cuDNN settings and seeded workers; roughly one third to one half again in training time |

Writes `checkpoints/<model>_<tag>_seed<seed>.pth`, the per-seed test and
validation predictions `results/probs_<tag>_<model>_seed<seed>.npz` and
`results/valprobs_<tag>_<model>_seed<seed>.npz`, and the aggregate
`results/ablation_<tag>_raw.csv`, `_summary.csv` and `.tex`. The tag is
`<task>_lf<fraction>_<scratch|imagenet|ssl>`. A run restricted with `--models` or
`--seeds` writes its aggregates under a name recording the restriction, so it
cannot overwrite a full run.

### `run_cross_source.py` — cross-source external validation

```bash
python run_cross_source.py --task pneumonia     # Table 5, Tables S1, S3
python run_cross_source.py --task tb            # Table 5, Tables S2, S4
```

Trains DenseNet-121, MobileNetV2, ShuffleNetV2 and DS-LiteDenseNet on the internal
corpus and scores each on its locked internal test split and on both external
hospitals (pneumonia: CheXpert and VinDr; tuberculosis: Montgomery and VinDr).
Keeps no checkpoints. Arguments: `--task`, `--seeds`, `--models`, `--epochs`,
`--deterministic`, as above.

Writes `results/crosssource_<task>_raw.csv` (one row per model, seed and cohort,
with AUROC, average precision, sensitivity, specificity, ECE and the per-seed
bootstrap interval), `_summary.csv` (mean and standard deviation across seeds) and
`.tex`. The average precision in Table 5 is the `auprc` column of the raw file,
filtered to VinDr and averaged over seeds.

`run_ablation.py` and `run_cross_source.py` train separate models. Their internal
AUROCs agree to within 0.002 on pneumonia but are not identical.

---

## Scoring existing checkpoints

### `eval_external.py` — external scoring without retraining

```bash
python eval_external.py --task pneumonia                     # Table S7 (both sites)
python eval_external.py --task pneumonia --source chexpert \
    --pos_label "Lung Opacity" \
    --chexpert_csv data/chexpert/CheXpert-v1.0-small/train.csv \
    --out_suffix _opacity                                    # Lung Opacity column, Table S14
python eval_external.py --task tb                            # third block of Table 5, Table S13
python eval_external.py --task pneumonia --tag pneumonia_lf0.05_scratch   # Table S6
python eval_external.py --task tb --tag tb_lf1.0_imagenet    # Table S8
# Table S12: set PER_IMAGE_NORM = False in config.py, then
python eval_external.py --task tb --out_suffix _nonorm       # and restore the setting afterwards
```

Scores the checkpoints written by `run_ablation.py` on the external cohorts.

| Argument | Default | Meaning |
|---|---|---|
| `--task` | `pneumonia` | `pneumonia` or `tb` |
| `--tag` | `<task>_lf1.0_scratch` | which checkpoints to score |
| `--models`, `--seeds` | all, `config.SEEDS` | restrict the run |
| `--source` | every site in the manifest | e.g. `chexpert` or `vindr` |
| `--pos_label`, `--chexpert_csv` | none | an alternative CheXpert positive column, scored against the same negatives |
| `--out_suffix` | empty | appended to every output name; must begin with an underscore |
| `--limit` | none | score only the first N images per source, half of each class, to check a run before the full cohort |
| `--batch`, `--workers` | `32`, `4` | decoding is the bottleneck; raise `--workers` to the number of physical cores |
| `--from_npz` | off | rebuild the CSVs from saved `extprobs_*.npz`, scoring nothing |

Writes the per-seed predictions
`results/extprobs_<tag>_<source>_<model>_seed<seed><suffix>.npz` and
`results/external_<tag><suffix>[_<sources>]_raw.csv` and `_summary.csv`. A full run
(no `--source`) writes every site into one file, distinguished by the
`eval_source` column; a run restricted with `--source` adds the site names to the
file name, so it cannot overwrite a full run.

### `border_occlusion.py` — occlusion control

```bash
python border_occlusion.py --task pneumonia --region border         # Table S18
python border_occlusion.py --task pneumonia --region centre         # Table S18
python border_occlusion.py --task pneumonia --source chexpert vindr # Table S18, external
python border_occlusion.py --task tb --region border                # Table S19
python border_occlusion.py --task tb --region centre                # Table S19
python border_occlusion.py --task tb --source montgomery vindr      # Table S19, external
python border_occlusion.py --task tb --region band_nonlung          # Table S21
python border_occlusion.py --task tb --region band_lung             # Table S21
python border_occlusion.py --task tb --region edge5_nonlung --fill region   # Table S22
python border_occlusion.py --task tb --region ring_nonlung  --fill region   # Table S22
python border_occlusion.py --task pneumonia --region edge5_nonlung --fill region   # Table S22, control
```

Re-scores existing checkpoints with one region of every image replaced by a
constant and reports the change in AUROC, paired image by image. AUROC is
threshold-free and both columns are scored on the same images. The fill is out of
distribution, so each change bounds the region's contribution from above rather
than measuring it.

| Argument | Default | Meaning |
|---|---|---|
| `--task` | `pneumonia` | `pneumonia` or `tb` |
| `--tag` | `<task>_lf1.0_scratch` | which checkpoints to score |
| `--models`, `--seeds` | all six, `config.SEEDS` | restrict the run |
| `--source` | internal test split | external cohorts instead |
| `--region` | `border` | see below |
| `--fill` | `retained` | `retained`: mean of everything kept (the fill of Tables S18–S21); `region`: the region's own mean, which destroys its structure and keeps its brightness |
| `--border` | `0.15` | band width; must leave the 158-pixel centre used by the Grad-CAM border share |
| `--edge` | `0.05` | width of the outer ring for the `edge5*` regions (11 px at 224) |
| `--batch`, `--workers`, `--device` | `32`, `4`, auto | loader and device |
| `--out` | none | optional CSV of the rows printed |

| `--region` | Replaced pixels | Share of the frame |
|---|---|---|
| `border` | outer 15% band | 0.50 |
| `centre` | the 158-pixel centre the band leaves | 0.50 |
| `band_nonlung` | band minus the lung mask | about 0.48 on Shenzhen |
| `band_lung` | band ∩ lung mask | about 0.02 on Shenzhen |
| `edge5` | outer 5% ring | 0.19 |
| `edge5_nonlung` | outer 5% ring minus the lung mask | about 0.19 |
| `ring_nonlung` | 5%–15% ring minus the lung mask | about 0.30 |

`edge5 ∪ ring = border` and `band_nonlung ∪ band_lung = border`. Each run prints
the replaced share of the frame first, then one line per model and cohort with
plain and occluded AUROC, their difference, the difference **per unit of
replaced area**, and the five per-seed pairs. Compare regions of different size by
the per-area figure. The `--out` CSV has the same rows with the columns `area`,
`delta_per_area`, `fill` and `edge`.

The lung-aware regions segment each image once and cache the packed masks in
`results/lungmask_<task>_<cohort>.npz`; the first pass over 2230 pneumonia images
takes several minutes. Everything a DataLoader worker needs is captured in the
parent process, so `--workers > 0` is safe on Windows. If a worker exits
unexpectedly on a first run, rerun the command or pass `--workers 2`.

---

## Analyses of saved outputs

### `paired_external.py` — paired statistics from saved per-seed AUROCs

```bash
python paired_external.py --files results/crosssource_pneumonia_raw.csv \
    results/external_pneumonia_lf1.0_scratch_chexpert_raw.csv \
    results/external_pneumonia_lf1.0_scratch_opacity_chexpert_raw.csv \
    results/external_pneumonia_lf1.0_scratch_vindr_raw.csv \
    results/crosssource_tb_raw.csv results/external_tb_lf1.0_scratch_raw.csv \
    --out results/paired_panels.csv
```

Reads any `*_raw.csv` written by `run_ablation.py`, `run_cross_source.py` or
`eval_external.py`, groups rows by `eval_source`, pairs each baseline with `--ref`
seed by seed, and reports the mean difference (baseline minus ours), the
two-sided paired *t*-test and the upper one-sided 95% bound. This is the
computation behind Tables S3, S4, S13, S14 and the rows of Table S15. Nothing is
retrained or re-scored.

| Argument | Default | Meaning |
|---|---|---|
| `--files` | required | CSVs or globs |
| `--ref` | `DSLiteDenseNet` | reference model |
| `--margin` | `0.02` | non-inferiority margin δ |
| `--tag` | none | keep only rows with this tag |
| `--out` | none | CSV of every row |

Output columns: `file, cohort, baseline, n_seeds, delta, p, bound, tcrit,
ni_shown, near_margin`. `tcrit` is the critical value actually used,
t(0.95, n_seeds − 1) = 2.132 for five seeds. `near_margin = 1` flags a bound
within 0.001 of δ, which the Supplementary tables print to four decimals. If a
per-site file such as `external_tb_lf1.0_scratch_vindr_raw.csv` is requested and
does not exist, the combined file written by a full `eval_external.py` run is
read and filtered to that site, and the substitution is printed.

### `paired_bootstrap.py` — non-inferiority with test-set uncertainty

```bash
python paired_bootstrap.py --task pneumonia --out results/paired_bootstrap_pneumonia.csv
python paired_bootstrap.py --task tb
```

The seed-level bound treats the test set as fixed. This script adds the
uncertainty of having sampled the test patients: for every baseline it reports the
seed-level bound, a bound from resampling patients within each seed, and a
two-level bound resampling both (Table S17, Section S20).

| Argument | Default | Meaning |
|---|---|---|
| `--task` | `pneumonia` | `pneumonia`, `tb` or `covid` |
| `--tag` | `<task>_lf1.0_scratch` | prefix of the `probs_*.npz` files |
| `--ref`, `--baselines`, `--seeds` | DS-LiteDenseNet, the other five, `config.SEEDS` | what is compared |
| `--margin`, `--alpha` | `0.02`, `0.05` | margin and one-sided level |
| `--n_boot`, `--boot_seed` | `2000`, `0` | resampling |
| `--out` | none | CSV of the table |

### `operating_point.py` — screening threshold and prior-shift calibration

```bash
# Section 4.5: sensitivity held at 0.90 on validation, carried to the locked test
python operating_point.py --probs "results/probs_*_lf1.0_scratch_*.npz" \
    --val_probs "results/valprobs_*_lf1.0_scratch_*.npz" --seeds 0 1 2 3 4

# Section 4.5 and Supplementary S8: prior-shift correction at each external site
python operating_point.py \
    --probs "results/extprobs_pneumonia_lf1.0_scratch_vindr_*.npz" \
    --val_probs "results/valprobs_pneumonia_lf1.0_scratch_*.npz" \
    --tag pneumonia_lf1.0_scratch --out results/operating_point_pneumonia_vindr.csv
```

Two questions answered from saved predictions. First, a screening operating
point: a threshold that holds sensitivity at `--target_sens` on the validation
split, carried unchanged to the file scored, with the specificity that survives.
Second, calibration under prior shift: the class-balanced sampler trains an even
prior, so shifting each logit by the difference in log prior odds removes exactly
that component of the external calibration error, and the script reports the ECE
before and after.

| Argument | Default | Meaning |
|---|---|---|
| `--probs` | required | glob over `.npz` files holding labels and probabilities |
| `--val_probs` | none | glob for the matching validation predictions; the threshold is fixed on these |
| `--target_sens` | `0.90` | sensitivity to hold |
| `--train_prior`, `--test_prior` | `0.5`, measured per file | priors for the correction |
| `--seeds`, `--tag` | none | restrict the files read |
| `--allow_test_threshold` | off | permit a threshold chosen on the scored file when no validation file matches (test-set reuse; an optimistic ceiling) |
| `--j_tolerance` | `0.02` | how much of Youden's *J* a stored external threshold may give up on the validation predictions before the file is reported as coming from different weights |
| `--out` | none | CSV |

### `gradcam_stats.py` — tests for the localisation measures

```bash
python gradcam_stats.py --seeds 0 1 2 3 4       # Table 6 and every Section 4.4 statistic
python gradcam_stats.py --task tb --model-a ShuffleNetV2 --model-b "" --seeds 0 1 2 3 4
python gradcam_stats.py --suffix _seed0          # within one seed
```

Consumes the per-image rows written by `run_gradcam.py` and reproduces Table 6 and
the tests of Section 4.4. Between models, on the same images, continuous measures
use a paired *t*-test with a Wilcoxon signed-rank test alongside, and the binary
pointing hit an exact McNemar test. Between classes within a model, Welch's
*t*-test with Mann–Whitney alongside. With `--seeds`, the across-seed tests treat
the training run as the unit and report class contrasts in excess of the lung
area actually available.

| Argument | Default | Meaning |
|---|---|---|
| `--task` | `both` | `pneumonia`, `tb` or `both`; a task with no CSVs for the requested models is skipped with a message |
| `--seeds` | none | aggregate over these seeds |
| `--seed_suffix` | `_seed{seed}` | suffix of the per-seed CSVs |
| `--suffix` | empty | read one suffixed file instead |
| `--model-a`, `--model-b` | `DSLiteDenseNet`, `DenseNet121` | paired tests report B minus A; pass `--model-b ""` for one model alone |
| `--out` | none | with `--seeds`, the per-seed table as CSV |

### `geometric_reference.py` — area references for Table 6

```bash
python geometric_reference.py            # both tasks
python geometric_reference.py --task tb
```

An energy share is interpretable only against what a map with no spatial
information would give, which is the area of the region. The border band is fixed
by construction (0.502 at 224 px, the centre being 158 px). The lung area depends
on the images and the segmenter, so it is measured on the same images the
Grad-CAM analysis scores, overall and by class. The script also reports how much
of the segmented lung lies inside the outer band (over the whole mask and in the
top strip alone) and how much of the band is lung, which Section S21 quotes (on
the Shenzhen test split the lung fills 0.036 of the band and 0.051 of the lung lies
inside it; on the pneumonia sample 0.047 and 0.069).

| Argument | Default | Meaning |
|---|---|---|
| `--task` | `both` | `pneumonia`, `tb` or `both` |
| `--n_per_class` | `120` | must match the `run_gradcam.py` call being described |

---

## Localisation

### `run_gradcam.py` — quantified Grad-CAM

```bash
for S in 0 1 2 3 4; do
  for t in pneumonia tb; do
    for m in DSLiteDenseNet DenseNet121 ShuffleNetV2; do
      python run_gradcam.py --task $t --model $m \
          --ckpt checkpoints/${m}_${t}_lf1.0_scratch_seed${S}.pth --out_suffix _seed${S}
    done
  done
done
```

Computes Grad-CAM at the final feature layer with respect to the positive class
and reports per image: `lung_energy` (share inside the lung mask), `bg_energy`
(share in the outer 15% band), and on RSNA positives `box_energy` (share inside
the radiologist box) and `pointing_hit` (whether the peak falls in a box). Also
saves the qualitative overlay panels of Figures 4 and 5 and Supplementary Fig. S1.

| Argument | Default | Meaning |
|---|---|---|
| `--task` | `pneumonia` | `pneumonia` or `tb` |
| `--ckpt` | required | checkpoint to explain |
| `--model` | `DSLiteDenseNet` | architecture of the checkpoint |
| `--n_per_class` | `120` | size of the fixed pneumonia sample per class |
| `--out_suffix` | empty | e.g. `_seed1`, so seeds do not overwrite one another |
| `--n_panels`, `--panel_px`, `--cam_alpha`, `--cam_floor` | 2, 512, 0.85, 0.15 | appearance of the overlay panels only |
| `--num_classes`, `--pos_class` | 2, 1 | output layout |

Writes `results/gradcam_<model>_<task><suffix>_raw.csv` and `_summary.csv`, and the
panels `figures/gradcam_<model>_<task><suffix>_panels.{pdf,png}` with a list of the
image ids shown. Figures 4 and 5 are the seed-0 panels copied under the unsuffixed
name. Without `torchxrayvision` the script still runs and writes the border and
box columns, but no lung columns.

---

## Calibration plots

### `plot_reliability.py`

```bash
python plot_reliability.py --task pneumonia    # Figure 6
python plot_reliability.py --task tb           # Supplementary Fig. S2
```

Reliability diagrams from the saved per-sample test predictions, pooling the named
seeds (confidence is the probability of the predicted class, following Guo et al.,
2017). Writes `figures/reliability_<tag>.{png,pdf}` and
`results/reliability_<tag>.csv`, the pooled ECE per model. Files matching the
pattern but outside `--seeds` are listed and skipped, and the script refuses to
write a table in which models were pooled over unequal numbers of files.
Arguments: `--task`, `--tag`, `--models`, `--n_bins` (15), `--seeds`.

### `export_reliability_curve.py`

```bash
python export_reliability_curve.py
```

A single model's pooled reliability curve as exact bin coordinates
(`reliability_curve_<model>_<task>.csv`) and a minimal vector panel (`.pdf`),
written to the working directory; the calibration curve of the graphical abstract
is traced from it. Arguments: `--model` (`DSLiteDenseNet`), `--tag`
(`pneumonia_lf1.0_scratch`), `--n_bins` (15), `--seeds`.

---

## Diagnostics and cost

### `diagnose_tb_domain.py`

```bash
python diagnose_tb_domain.py        # Supplementary Fig. S3, Section S15
```

Separates an input-level difference between Shenzhen and Montgomery (bit depth,
intensity, contrast), which preprocessing can correct, from a difference in the
pathology or population, which it cannot. Bit depth is read directly from disk;
intensity summaries are printed at three stages of the pipeline and a side-by-side
panel is saved to `figures/diag_tb_domain_panel.{pdf,png}`.

### `bench_inference.py`

```bash
python bench_inference.py --threads 24
```

Latency and activation cost of one forward pass at 224×224, batch 1, on CPU by
default; no data or checkpoints are needed. Reports the median of `--runs` timed
passes after `--warmup` passes, and the total activation data written by a pass.
Writes `results/inference_cost.csv`. Arguments: `--device` (`cpu`), `--batch` (1),
`--runs` (50), `--warmup` (10), `--threads`, `--models`.

The manuscript's CPU column is the shortest of eighteen repetitions of the script;
a single run varies by 22–37% on an idle machine and is not meant to reproduce it.
What reproduces is the ratios between models, which the manuscript argues from.
Parameter, activation and weight figures depend only on the architecture and the
input size and reproduce exactly. Latency on mobile runtimes with optimised
depthwise kernels may order the models differently.

---

## Figures

| Script | Figure | Output |
|---|---|---|
| `paper_figures/conv_comparison_standard.py` | Figure 1(a) | `figures/Figure_comparison_calculation_a.{pdf,png}` |
| `paper_figures/conv_comparison_separable.py` | Figure 1(b) | `figures/Figure_comparison_calculation_b.{pdf,png}` |
| `paper_figures/dense_block_diagram.py` | Figure 2 | `figures/Figure_Dense_Block_Structure.{pdf,png}` |
| `paper_figures/network_architecture.py` | Figure 3 | `figures/Network_Architecture.{pdf,png}` |
| `paper_figures/label_efficiency.py` | Figure 7 | `figures/label_efficiency.{pdf,png}` |

The schematics depend on nothing but matplotlib; `label_efficiency.py` reads the
per-fraction summaries written by `run_ablation.py --label_fraction`, so the
figure cannot drift from the results it plots. Every script accepts `--out` for
the output directory. The two panels of Figure 1 are combined into
`Figure_comparison_calculation.pdf` by hand. All figure scripts embed TrueType
fonts (`pdf.fonttype = 42`).

---

## Supplementary numbering

| Content | Section | Tables |
|---|---|---|
| Grad-CAM panels for DenseNet-121 | S1 | Fig. S1 |
| Per-seed cross-source results | S2 | S1, S2 |
| Paired comparisons and non-inferiority | S3 | S3, S4 |
| Tuberculosis calibration | S4 | Fig. S2 |
| Operating points, tuberculosis and COVID-19 | S5 | S5 |
| Reproducibility and provenance | S6 | — |
| Grad-CAM normalisation | S7 | — |
| Prior-shift correction | S8 | — |
| CPU benchmark variation | S9 | — |
| External pneumonia, ablation checkpoints | S10 | S6, S7 |
| ImageNet initialisation, tuberculosis / pneumonia | S11 / S12 | S8 / S9 |
| Tuberculosis run-to-run variation | S13 | S10 |
| Label-efficiency calibration | S14 | S11 |
| Tuberculosis input-level difference | S15 | Fig. S3, S12 |
| Paired external comparisons, tuberculosis / pneumonia | S16 / S17 | S13 / S14 |
| Every comparison with *p* < 0.05 | S18 | S15 |
| COVID-19 arm | S19 | S16 |
| Two-level bootstrap | S20 | S17 |
| Occlusion control | S21 | S18–S22 |
| Notes to the main text | S22 | — |
