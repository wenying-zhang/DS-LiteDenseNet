# Script reference

Run order for a full reproduction is `data_manifests.py`, then the two training
drivers, then the analysis scripts. Everything below assumes the working
directory is the repository root.

Shared conventions:

- Dataset roots, seeds and hyper-parameters come from `config.py`. No script
  takes a dataset path on the command line. The flags that do take a path name
  something `config.py` cannot know in advance: a specific checkpoint
  (`--ckpt`), a glob over saved predictions (`--probs`), an alternative label
  file (`--chexpert_csv`), an output directory for the figure scripts (`--out`,
  in `paper_figures/`), an input directory for `label_efficiency.py`
  (`--results`), and a destination CSV for `operating_point.py` (`--out`).
- Training scripts write a `*_raw.csv` with one row per model and seed, a
  `*_summary.csv` with the aggregate, and a `.tex` fragment, all under
  `results/`.
- The training loader drops its last batch only when that batch would hold a
  single image, which `nn.BatchNorm2d` cannot compute a variance over. No
  configuration in the manuscript leaves a remainder of one, so every reported
  run is unaffected; a `--label_fraction` that does hit it prints a line saying
  so.
- Figures go to `figures/`, checkpoints to `checkpoints/`.
- The decision threshold is always fixed on the validation split by Youden's
  *J* and applied unchanged to the test split.

---

## `config.py`

Central settings. Nothing to run.

Edit `DATA_ROOT`, or individual entries in `RAW_DATA`, to match the layout of
the corpora on disk. The other settings worth knowing about:

| Name | Default | Effect |
|------|---------|--------|
| `USE_VINDR_EXTERNAL` | `True` | Include VinDr-CXR as a second external site. Requires the adult release. |
| `USE_CLAHE` | `True` | Contrast-limited adaptive histogram equalisation on every image. |
| `PER_IMAGE_NORM` | `True` | Per-image standardisation instead of fixed ImageNet statistics. |
| `SEEDS` | `[0,1,2,3,4]` | Seeds used for every mean, standard deviation and paired test. |
| `IMG_SIZE` | `224` | Network input, in pixels. Latency, activation cost and the Grad-CAM mask resolution all follow it. Not settable from any command line. |
| `EPOCHS` / `EARLY_PATIENCE` | `100` / `15` | Budget and early-stopping patience on validation AUROC. |

Changing either preprocessing flag invalidates results computed under the other
setting; the pipeline must be re-run as a whole.

---

## `data_manifests.py`

Builds the manifests that every other script reads. Run once after the data is
in place.

```bash
python data_manifests.py
```

Writes to `manifests/`:

| File | Content |
|------|---------|
| `pneumonia_internal.csv` | RSNA, split into train / val / test |
| `pneumonia_external.csv` | CheXpert and VinDr, `split == 'external'` |
| `tb_internal.csv` | Shenzhen, split into train / val / test |
| `tb_external.csv` | Montgomery and VinDr |
| `covid_internal.csv` | COVIDGR-1.0, single source, no external arm |

Columns: `image_id, path, file_type, label, patient_id, source, split`.

Points to note:

- The internal split is patient-wise and stratified at 70/15/15, so no patient
  appears in more than one partition.
- RSNA's ambiguous `No Lung Opacity / Not Normal` class is dropped, leaving
  `Lung Opacity` as positive and `Normal` as negative.
- VinDr labels are aggregated by strict majority, 2 of 3 readers on the raw
  training file. Images without majority agreement are discarded.
- Rows whose image file is missing from disk are dropped up front, with a count
  and three example paths printed, rather than failing later in the DataLoader.

---

## `run_ablation.py`

Architecture ablation and comparison against the lightweight baselines, on the
locked internal test split.

```bash
python run_ablation.py --task pneumonia
python run_ablation.py --task tb
python run_ablation.py --task covid
python run_ablation.py --task pneumonia --label_fraction 0.1
python run_ablation.py --task tb --imagenet        # Supplementary S12
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--task` | `pneumonia` | `pneumonia`, `tb` or `covid` |
| `--seeds` | `config.SEEDS` | Seeds to train |
| `--models` | all six | Subset of `config.ABLATION_MODELS` |
| `--epochs` | `config.EPOCHS` | Override the epoch budget |
| `--label_fraction` | `1.0` | Stratified fraction of the training split |
| `--imagenet` | off | Initialise the three torchvision baselines (DenseNet121, MobileNetV2, ShuffleNetV2) from ImageNet weights; the three architectures defined here have none and stay from scratch, which the script says per model. Tags the run `_imagenet`, so it cannot overwrite the scratch results. Mutually exclusive with `--pretrained` |
| `--pretrained` | `None` | Checkpoint for a self-supervised encoder. Not used by any result in the manuscript, which trains every model from scratch; the flag and its `_ssl` tag exist so that a pretrained comparison can be added without disturbing the scratch runs. Only encoder weights are loaded: every classification head is excluded by name, `classifier` for the DenseNet variants and MobileNetV2 and `fc` for ShuffleNetV2, so a checkpoint of the same class count cannot carry a fitted head into a run reported as encoder-only |
| `--ni_margin` | `0.02` | AUROC margin for the non-inferiority test |
| `--from_npz` | off | Rebuild the three aggregate files from the per-seed `.npz` in `results/`, training nothing. See below |
| `--deterministic` | off | Disable cuDNN autotuning, request deterministic algorithms, and set `CUBLAS_WORKSPACE_CONFIG` before the CUDA context is created. Costs roughly a third to a half again in time; see the reproducibility section of the README |

Outputs, with `tag = <task>_lf<fraction>_<scratch|ssl|imagenet>`:

- `results/ablation_<tag>_raw.csv` — one row per model and seed
- `results/ablation_<tag>_summary.csv` — aggregate, plus `p_paired_vs_ours`,
  `p_delong_vs_ours` and the non-inferiority verdict column
- `results/ablation_<tag>.tex` — the same columns as Table 3 of the
  manuscript (AUROC, Acc, Sens, Spec, F1, ECE, parameters, and the
  non-inferiority verdict with its gap and paired *p*), so the fragment can be
  used as-is rather than completed by hand
- `results/probs_<tag>_<model>_seed<seed>.npz` — per-sample test
  probabilities, which is what lets the calibration figures be produced without
  retraining
- `results/valprobs_<tag>_<model>_seed<seed>.npz` — the same for the
  validation split, from which `operating_point.py --val_probs` fixes its
  sensitivity-constrained threshold
- `checkpoints/<model>_<tag>_seed<seed>.pth`, one per model and seed

A run restricted with `--models` or `--seeds` writes its three aggregate files
as `ablation_<tag>_<models>[_seeds<list>]_*`, so rerunning one model cannot
replace the six-model summary of a full run under the same tag. The per-seed
`.npz` files and checkpoints are named by model and seed already and are
replaced only for the models that ran.

With `--imagenet`, DenseNet121, MobileNetV2 and ShuffleNetV2 are initialised
from the torchvision ImageNet checkpoints. DenseNet121 is the repository's own
`DenseNetVariant`, whose encoder is layer-for-layer that of torchvision's
`densenet121` under different parameter names; `engine.load_imagenet_encoder`
translates the names and refuses to start training unless every encoder tensor
has been loaded. Each run prints the count: 725 for DenseNet121, 312 for
MobileNetV2 and 336 for ShuffleNetV2, with only the classification head left at
random initialisation. An earlier version matched names directly, loaded
nothing into DenseNet121, and trained it from scratch under the `_imagenet`
tag; its log line read `loaded 0 ImageNet tensors`.

The non-inferiority column answers: is DS-LiteDenseNet within `--ni_margin`
AUROC of the model in this row, judged by the upper one-sided 95 % bound on the
paired per-seed difference? A single-seed run yields `--` rather than an error.

Produces manuscript Tables 2, 3, 4 and 7, and Supplementary Table S5 when run
with `--task tb` and `--task covid`. The `probs_*.npz` files it writes are the
input to `plot_reliability.py`, `export_reliability_curve.py` and
`operating_point.py`; `paper_figures/label_efficiency.py` reads the summary
CSVs instead, since a curve of mean AUROC needs no per-sample probabilities.

---

## `run_cross_source.py`

External validation. Trains on the internal corpus and evaluates on the locked
internal test split and on both external hospitals.

```bash
python run_cross_source.py --task pneumonia
python run_cross_source.py --task tb
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--task` | `pneumonia` | `pneumonia` or `tb`; COVIDGR is single-source and has no external arm |
| `--seeds` | `config.SEEDS`, i.e. all five | Reported results use all five |
| `--models` | the four in `MODELS` | DenseNet-121, MobileNetV2, ShuffleNetV2 and DS-LiteDenseNet; the two intermediate ablation variants are not retrained per source |
| `--epochs` | `config.EPOCHS` | Override the epoch budget |
| `--deterministic` | off | As for `run_ablation.py` |

Outputs `results/crosssource_<task>_{raw,summary}.csv` and
`results/crosssource_<task>.tex`. The `.tex` fragment carries the AUROC columns
and the average-precision column at the low-prevalence external site, matching
Table 5 of the manuscript.

The raw CSV carries every metric per model, seed and evaluation source: `auroc`,
`auprc`, `acc`, `bacc`, `sens`, `spec`, `f1`, `ece`, the bootstrap bounds
`auroc_lo` and `auroc_hi`, and `n`. The summary aggregates AUROC only, so the
average precision and calibration figures quoted in the manuscript are read from
the raw file.

Produces manuscript Table 5 and Supplementary Tables S1–S4.

---

## `run_gradcam.py`

Quantified Grad-CAM localisation, plus the qualitative overlay panels.

```bash
# one checkpoint
python run_gradcam.py --task pneumonia --model DSLiteDenseNet \
    --ckpt checkpoints/DSLiteDenseNet_pneumonia_lf1.0_scratch_seed0.pth --out_suffix _seed0

# Table 6: both models, both tasks, all five seeds
for S in 0 1 2 3 4; do
  for t in pneumonia tb; do
    for m in DSLiteDenseNet DenseNet121; do
      python run_gradcam.py --task $t --model $m \
          --ckpt checkpoints/${m}_${t}_lf1.0_scratch_seed${S}.pth --out_suffix _seed${S}
    done
  done
done
python gradcam_stats.py --seeds 0 1 2 3 4
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--task` | `pneumonia` | Task whose manifest and masks to use |
| `--ckpt` | required | Checkpoint to explain |
| `--model` | `DSLiteDenseNet` | Architecture matching the checkpoint |
| `--num_classes` | `2` | For checkpoints with a different head |
| `--n_per_class` | 120 | Images scored per class, taken in manifest order. A split with fewer than this contributes all it has: the tuberculosis test split holds about 50 positives, so the TB row of Table 6 is scored on those |
| `--n_panels` | 2 | Qualitative overlays drawn in the figure |
| `--pos_class` | 1 | Class index explained by the CAM |
| `--cam_alpha`, `--cam_floor` | 0.85, 0.15 | Overlay opacity at the CAM peak and where the CAM is zero |
| `--out_suffix` | empty | Appended to every output name so that repeating the analysis across seeds does not overwrite the previous one. The leading underscore is part of the value, e.g. `_seed1`. `gradcam_stats.py --suffix` reads the same names back |
| `--panel_px` | 512 | Pixel size of each panel tile |

Outputs, for `<model>`, `<task>` and `<suffix>` from `--out_suffix`:

```
results/gradcam_<model>_<task><suffix>_raw.csv         one row per scored image
results/gradcam_<model>_<task><suffix>_summary.csv     means by class
figures/gradcam_<model>_<task><suffix>_panels.pdf      the overlay panel
figures/gradcam_<model>_<task><suffix>_panels.png      raster fallback
figures/gradcam_<model>_<task><suffix>_panels_ids.txt  which case is in which row
```

The manuscript's Figures 4 and 5 and Supplementary Figure S1 are the `_seed0`
panels; Table 6 is computed from all five `_seed<N>` raw CSVs by
`gradcam_stats.py --seeds`.

The panel figure carries no image identifiers, so that the radiographs are not
captioned with them; the sidecar `_ids.txt` records which case each row shows,
for anyone who needs to trace a panel back to a patient.

The raw CSV is the input to `gradcam_stats.py` and carries the `lung_area`
column that `geometric_reference.py` also measures. Bounding-box coordinates
are read from the raw RSNA label CSV named in `config.py`, not from the
generated manifest, which stores no box columns.

Reported quantities, all as a share of total CAM mass:

| Metric | Definition | Direction |
|--------|------------|-----------|
| `lung_energy` | inside the lung mask | higher is better |
| `bg_energy` | in the outer 15 % border | lower is better |
| `box_energy` | inside the radiologist box, RSNA positives only | higher is better |
| `pointing_hit` | CAM peak falls inside a box, RSNA positives only | higher is better |

Read these against the area baselines: the border covers about half the pixels
and the lungs about a third.

Lung masks come from the ChestX-Det PSPNet via `lung_seg.py`, except on
Montgomery, where the shipped manual masks are used. The target layer is
`final_bn` for the DenseNet variants and the last convolutional block for the
baselines; see `models.get_target_layer`.

Produces the per-seed inputs to Table 6, Figures 4 and 5, and Supplementary
Figure S1.

---

## `lung_seg.py`

Library module, not a script. Wraps the ChestX-Det PSPNet and takes the union of
the left and right lung channels; `montgomery_mask` returns the shipped manual
mask instead where one exists.

Segmenter: Lian et al., *IEEE Trans. Med. Imaging* 40(8):2042–2052, 2021,
distributed through TorchXRayVision (Cohen et al., MIDL 2022). First use
downloads about 260 MB of weights and needs network access.

---

## `plot_reliability.py`

Reliability diagrams and per-model ECE, from the saved probabilities. No GPU and
no retraining.

```bash
python plot_reliability.py --task pneumonia
python plot_reliability.py --task tb
python plot_reliability.py --tag pneumonia_lf0.1_scratch
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--task` | `pneumonia` | Shorthand for the default tag |
| `--tag` | derived | Explicit tag, for label-fraction or SSL runs |
| `--models` | all six | Subset to plot |
| `--n_bins` | 15 | Confidence bins, matching the manuscript's ECE |
| `--seeds` | `config.SEEDS` | Seeds to pool; must match the `run_ablation.py` call that wrote the files |

Writes `figures/reliability_<tag>.png` and `figures/reliability_<tag>.pdf`,
plus `results/reliability_<tag>.csv`. The manuscript includes the PDF.

Within each panel the markers sit at the bin's mean confidence, which is the
value the bin accuracy is compared against, while the bars sit at the bin's
geometric centre so that they tile the axis. Centring a bar of width
`1/--n_bins` on the mean confidence instead pushes the top bar past 1.0 and
overlaps its neighbours lower down.

The seeds are named rather than globbed. `results/` accumulates probability files
from every run that ever used a tag, including short diagnostic runs and aborted
sweeps, and a `seed*` glob pools all of them without saying so: one model is then
summarised over more predictions than the models it is compared against. Files
matching the pattern but outside `--seeds` are listed and skipped, and the script
exits rather than write a table whose models were pooled over unequal counts. If
that happens, either re-run the ablation for the short-changed models or pass
`--seeds` to name the intended set.

Each panel pools the predictions of the named seeds so that every bin is populated
densely enough to be stable. Because ECE is a non-linear functional whose
finite-sample estimate carries a positive bias that shrinks as bin occupancy
grows, the pooled value need not equal the mean of the per-seed values reported
in the ablation tables, and is generally slightly smaller. Both are correct;
they are different summaries of the same predictions.

Produces manuscript Figure 6 and Supplementary Figure S2.

---

## `export_reliability_curve.py`

Exports one reliability curve as vector coordinates plus a small standalone PDF,
for use as a single panel. `plot_reliability.py` is the script for the
multi-model figure.

```bash
python export_reliability_curve.py
python export_reliability_curve.py --model DenseNet121
python export_reliability_curve.py --seeds 0 1 2 3 4
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--model` | `DSLiteDenseNet` | Model whose curve to export |
| `--tag` | `pneumonia_lf1.0_scratch` | Run tag of the probability files |
| `--n_bins` | `15` | Equal-width confidence bins. Matches `stats.py`, `plot_reliability.py` and the manuscript; changing it changes the curve's shape, not the model. Spelled `--n_bins`, as in `plot_reliability.py`. `--bins` is accepted as an alias, so commands written against earlier revisions still run |
| `--seeds` | `config.SEEDS` | Seeds to pool; must match the `run_ablation.py` call that wrote the files |

Writes `reliability_curve_<model>_<task>.csv` and the matching PDF to the
working directory, where <task> is taken from `--tag`.

> This script writes to the working directory. Every other plotting script
> writes to `figures/` (`config.FIG_DIR`), while the manuscript sources its
> figures from `figs/`; copy or symlink the PDFs across after regenerating.
> Nothing in the code depends on the manuscript's path.

---

## `bench_inference.py`

Measures what a forward pass costs: median latency and peak activation memory.
Needs no data, no checkpoints and no training, since the cost of a forward pass
depends on the architecture and input size rather than on the weights.

```bash
python bench_inference.py                 # CPU, batch 1, all six models
python bench_inference.py --threads 4     # pin the core budget
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--device` | `cpu` | `cpu` or `cuda`; CPU is the point-of-care case |
| `--batch` | 1 | 1 is one radiograph at a time |
| `--runs` | 50 | Timed passes; the median is reported |
| `--warmup` | 10 | Untimed passes before timing starts |
| `--threads` | torch default | Pin this so the timing is reproducible |
| `--models` | all six | Subset to measure |

*Note.* The input is `config.IMG_SIZE`, 224x224, and is not settable from the
command line. Latency and activation cost both scale with it, so it is the
figure to quote alongside any timing; change it in `config.py` if a different
input is wanted, and be aware that doing so also changes what every training
script feeds the network.

Table 2 reports parameters and FLOPs alongside these figures; neither is what an
offline device is limited by. Note that this script does not compute FLOPs and
its CSV has no such column: the FLOP counts come from `models.count_flops_g`,
which `run_ablation.py` calls and folds into its summary CSV. Latency is set by memory traffic as much as arithmetic, and dense
concatenation moves a lot of memory per FLOP. Writes
`results/inference_cost.csv`.

The parameter, activation and weight columns depend only on the architecture and
the input size, so they reproduce exactly. The latency column does not, and the
size of the effect is easy to underestimate: eighteen consecutive runs of this
script on the same idle 24-thread processor gave DenseNet-121 medians from 39.4
to 56.3 ms, and every model varied by 22 to 37 per cent of its own median. The
file is also overwritten by whichever run finished last, so it holds one sample
rather than a summary.

Table 2 of the manuscript is therefore the per-model *minimum* over eighteen
repetitions, on the reasoning that contention only ever adds time. To compare
against it, run the script several times and take the minimum; to quote a single
run, quote the thread count with it and read the ratios between models rather
than the absolute values.

---

## `eval_external.py`

Scores checkpoints that already exist on an external cohort, so questions of the
form "how would these weights behave at another hospital" need no retraining.

```bash
# the 5% label fraction, evaluated externally
python eval_external.py --task pneumonia --tag pneumonia_lf0.05_scratch
python eval_external.py --task pneumonia --tag pneumonia_lf1.0_scratch

# ImageNet-initialised checkpoints (Supplementary S12)
python eval_external.py --task tb --tag tb_lf1.0_imagenet

# CheXpert scored against its own Lung Opacity column instead of Pneumonia
python eval_external.py --task pneumonia --source chexpert \
       --pos_label "Lung Opacity" --chexpert_csv path/to/labels.csv \
       --out_suffix _opacity
```

Pass `--out_suffix` whenever the label definition changes, or the second run
overwrites the first: the output names depend on the tag, not on the label.

| Argument | Default | Meaning |
|----------|---------|---------|
| `--task` | `pneumonia` | Task whose manifests and checkpoints to use |
| `--tag` | `<task>_lf1.0_scratch` | Checkpoint tag, e.g. `pneumonia_lf0.05_scratch` |
| `--models` | all six | Subset to score |
| `--seeds` | `config.SEEDS` | Seeds to score |
| `--source` | all in the manifest | External sources to score |
| `--pos_label` | none | Alternative positive column, e.g. `"Lung Opacity"` |
| `--chexpert_csv` | none | CheXpert label CSV; required with `--pos_label` |
| `--batch` | 32 | Inference batch size |
| `--workers` | 4 | Decoding, not the network, is the bottleneck |
| `--limit` | none | Score only the first this-many images per source, half of each class. A smoke test before the full cohort. It alters the prevalence, so AP, ECE and any precision-based figure stop being comparable with a full run; sensitivity and specificity do not depend on prevalence, but are then estimated on a small head of each class rather than on the cohort |
| `--out_suffix` | empty | Appended to output filenames, to keep two label definitions apart |
| `--from_npz` | off | Rebuild the two CSVs from the per-seed `.npz` files already in `results/`, scoring nothing. See below |

The threshold is taken from the internal validation split and carried unchanged,
as in the main experiments.

Outputs, with `<suffix>` from `--out_suffix`:

```
results/external_<tag><suffix>[_<sources>]_raw.csv       one row per model, seed and source
results/external_<tag><suffix>[_<sources>]_summary.csv   mean and std over seeds
results/extprobs_<tag>_<source>_<model>_seed<seed><suffix>.npz
```

`<sources>` appears only when `--source` restricts the run, and is the sources
joined by `-`; a `--models` or `--seeds` subset appends the models joined by
`-` and `_seeds<list>` in the same way, so rescoring one model does not replace
the CSVs holding all six: a full run writes `external_pneumonia_lf1.0_scratch_raw.csv`
holding every site, distinguished by the `eval_source` column, while
`--source vindr` writes `external_pneumonia_lf1.0_scratch_vindr_raw.csv`. The
`.npz` names have always carried the source; these two CSVs did not, so a run on
one site silently overwrote the CSVs of an earlier run on another. Only the CSVs
are affected — the probability files were never at risk.

The `.npz` files hold `y`, `prob` and the threshold, and are what
`operating_point.py` reads. That script parses the seed from the end of the
name and accepts an `--out_suffix` after it provided the character following the
underscore is a letter, so `_opacity` and `_lung_opacity` are read and a stray
`_1` from an aborted run is not. The suffix is kept in the grouping key there,
so two label definitions are never pooled into one mean, but it is left out of
the key that finds the validation predictions: those belong to the training
run, which is the same whichever label the external cohort is scored against.

**What `--pos_label` changes, and what it does not.** The negatives are the
manifest's No Finding cases in both runs. The positives are read from
`--chexpert_csv` over the whole label file, not selected out of the manifest.
That distinction is the point of the option: the manifest holds the pneumonia
cohort, which is Pneumonia positives plus No Finding negatives, so any case
carrying the alternative label but neither of those is simply absent from it.
Filtering inside the manifest would yield Lung Opacity *and* Pneumonia — a
pruned pneumonia cohort rather than a second label definition — and the
difference between the two runs would not be the quantity the comparison is
for. Positive prevalence therefore differs between the two runs, so compare
AUROC across the pair and not AP or ECE.

**Recovering an interrupted run (`--from_npz`).** The probability files are
written as each pass finishes, the two CSVs only at the end, so a run that dies
part-way through leaves its completed work on disk in a form nothing reads.
`--from_npz` rebuilds the CSVs from those files: each holds the labels, the
probabilities and the threshold that was used, which is every column the CSVs
carry. It scores nothing and takes seconds.

```bash
python eval_external.py --task pneumonia --tag pneumonia_lf1.0_imagenet --from_npz
```

It reads whatever matches the tag, so files from several invocations are
combined into one CSV, and a model or seed with no file is reported and skipped.
Restrict with `--models`, `--seeds`, `--source` and `--out_suffix` as usual; note
that those restrictions then also name the output file, so pass none of them to
write the plain, complete CSV.

Absolute values will not reproduce Table 5, which comes from its own training
run. Compare within one invocation, or between two that differ in one variable.

---

## `operating_point.py`

Two analyses over any saved probability file, with no retraining.

```bash
# Section 4.5: sensitivity held at 0.90 by a threshold fixed on validation
python operating_point.py --probs "results/probs_*_lf1.0_scratch_*.npz" \
    --val_probs "results/valprobs_*_lf1.0_scratch_*.npz" --seeds 0 1 2 3 4
# an external cohort, thresholds from the internal validation split of the same run
python operating_point.py --probs "results/extprobs_*vindr*.npz" \
    --val_probs "results/valprobs_*.npz" --tag tb_lf1.0_scratch
# prior-shift calibration only (Supplementary S8); needs no validation file
python operating_point.py --probs "results/extprobs_*vindr*.npz" --train_prior 0.5
```

The first holds sensitivity at a target and reports the surviving specificity,
which is the operating point screening guidance asks for and which Youden's *J*
does not give. The second removes the arithmetic part of the external
calibration error by shifting the logit by the log prior ratio between the
class-balanced training prior and the evaluation prevalence; whatever error
survives is attributable to transfer rather than to prior shift.

| Argument | Default | Meaning |
|----------|---------|---------|
| `--probs` | required | Glob over `.npz` files holding `y` and `prob` |
| `--val_probs` | `None` | Glob for the `valprobs_*.npz` written by `run_ablation.py`. The sensitivity-constrained threshold is fixed on these and carried unchanged to the split being scored; `thr_from` reads `validation`. Without it the sensitivity and specificity columns are left empty and only the calibration columns are computed. With it, a scored file that has no validation counterpart stops the run and is named |
| `--j_tolerance` | 0.02 | How much of Youden's *J* the threshold stored in an external file may give up on the matching validation predictions before that file is reported as scored with different weights |
| `--allow_test_threshold` | off | Permit choosing the threshold on the predictions being scored when no validation file matches. That is test-set reuse and gives an optimistic ceiling rather than an estimate; `thr_from` then reads `test (reused)`. No result in the manuscript uses it |
| `--target_sens` | 0.90 | Sensitivity to hold |
| `--train_prior` | 0.50 | Positive rate the sampler presented in training |
| `--test_prior` | measured | Positive rate at evaluation |
| `--seeds` | all | Restrict to these seeds; other files are skipped by name |
| `--tag` | all | Restrict to one run tag |
| `--out` | none | Write the per-file table to CSV |

Pass `--tag` when `results/` holds more than one run: files from different tasks
and label fractions must not be averaged together, and the script warns when
several tags match.

The validation files are matched to each scored file by model, seed and
training-run tag. An `--out_suffix` from `eval_external.py` is not part of that
match: it names a way of scoring the checkpoints, not a different training run.
Files written by `eval_external.py` also carry the Youden threshold it fixed on
the validation split, which is checked against the matching `valprobs` file. The
check is not a comparison of the two thresholds. Youden's *J* is often nearly
flat around its maximum, so a difference of $10^{-5}$ in one probability -- the
scale at which two executions of the same command differ when one enabled
deterministic kernels and the other did not -- moves the argmax across a
plateau, and the two numbers can differ by several hundredths while meaning the
same thing. What a checkpoint mismatch does produce is a threshold that is *bad*
for these predictions, so the script scores the stored threshold on the
validation predictions and reports how much of the achievable *J* it gives up.
Files above `--j_tolerance` are named; they should be rescored against the
current checkpoints.

---

## `geometric_reference.py`

Measures the two references that Table 6's energy shares are read against. A
Grad-CAM map carrying no spatial information integrates, over any region, to
that region's area fraction, so the area fraction is the value an energy share
has to beat.

```bash
python geometric_reference.py
python geometric_reference.py --task tb
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--task` | `both` | `pneumonia`, `tb`, or both |
| `--n_per_class` | 120 | Must match the `run_gradcam.py` call being described, since the reference is only valid for the images that run actually scored |

Prints the pooled lung, box and border references, and beneath the lung figure
the same quantity split by class. The split matters: the pooled figure is the
reference for a single share, but the reference for a *difference* between
classes is the difference in area. On RSNA the segmented lung is smaller on the
positives ($0.330$ against $0.280$) because consolidation obscures the boundary;
on Shenzhen it is larger ($0.298$ against $0.348$). An uninformative map loses
$0.050$ of lung energy between the pneumonia classes and gains $0.050$ between
the tuberculosis ones, which is the null each $\Delta$ in Table 6 is read
against.

The border reference is fixed by construction: `border_mask` cuts
`int(size * 0.15)` pixels from each edge, so at 224 px the centre is 158 px and
the border covers `1 - (158/224)**2 = 0.502`. The integer floor matters — the
idealised `1 - 0.7**2` would give 0.510.

The lung reference is not fixed. It depends on the images and on the segmenter,
and must be measured on the same images the Grad-CAM analysis scores; the
Shenzhen reference is not the RSNA one. This script reproduces
`run_gradcam.py`'s selection exactly (test split, first `--n_per_class` of each
label) and reports the mean mask area over it. Prints only; writes nothing.

`run_gradcam.py` also records a `lung_area` column per image, so a run's
reference can be recovered from its own raw CSV without re-segmenting.

---

## `gradcam_stats.py`

Recomputes every statistic quoted in Section 4.4, and every cell of Table 6,
from the raw CSVs that `run_gradcam.py` writes, so no value in that section is
derived by hand.

```bash
python gradcam_stats.py --seeds 0 1 2 3 4            # Table 6 and the across-seed tests
python gradcam_stats.py --task tb --suffix _seed0    # image-level tests on one seed
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--task` | `both` | `pneumonia`, `tb`, or both |
| `--model-a` | `DSLiteDenseNet` | Compact model; paired tests report B minus A |
| `--suffix` | empty | Read `gradcam_<model>_<task><suffix>_raw.csv` instead of the unsuffixed file, matching `run_gradcam.py --out_suffix` |
| `--seeds` | none | Aggregate over these seeds, reading one raw CSV per seed; see below |
| `--seed_suffix` | `_seed{seed}` | Name of each seed's suffix, with `{seed}` replaced |
| `--out` | none | With `--seeds`, write the per-seed table to this CSV |
| `--model-b` | `DenseNet121` | Baseline; pass an empty string to skip pairing |

Between models it pairs the annotated positives by `image_id` and reports a
paired *t*-test with a Wilcoxon signed-rank test beside it, plus an exact
McNemar test on the pointing hit. Between classes it reports an unpaired Welch
test with Mann-Whitney beside it and, where the contrast is not significant, the
95% interval and the effect the sample would detect at 80% power. Reads only;
writes a CSV only when `--out` is given.

After the lung-energy contrast it prints two further lines: the mean segmented
lung area in each class, and the contrast recomputed as lung energy *in excess of
that area*. A map with no spatial information integrates to exactly the mask's
area fraction, so when the mask is not the same size in both classes the raw
contrast credits or penalises the model for the anatomy. On tuberculosis the
segmented lung is larger on the positives, so the correction shrinks the
contrast; on pneumonia it is smaller on the positives, so the correction
enlarges it. Report the excess line
alongside the raw one.

**Across seeds (`--seeds`).** The training seed is the unit. Each seed's raw
CSV is reduced to its class means; the Table 6 cells are the means of those over
seeds. The lung energy in excess of mask area is tested against zero with a
one-sample *t*-test over the seed-level contrasts, with the number of seeds in
which it is positive; the two architectures are compared with a paired *t*-test
over seeds on positive-case lung energy, box energy, pointing, the excess
contrast and positive-case border energy. The image-level Welch and
Mann-Whitney tests are still run within every seed, and their largest *p*-value
over seeds is printed, which is how "in every seed" is reported in the text. The
lung area of each image does not depend on the model or seed, so the script
warns if it differs between seeds: that would mean the seeds scored different
images.

Columns that `run_gradcam.py` did not write are skipped rather than raising:
`box_energy` and `pointing_hit` exist for annotated RSNA positives only, so on
tuberculosis they are absent from the CSV entirely and the corresponding tests
are reported as not measured.

---

## `diagnose_tb_domain.py`

Separates an input-level mismatch from genuine domain shift on the TB task.

```bash
python diagnose_tb_domain.py
```

Reads pixel bit depth straight from disk, independently of `datasets.py`, so the
diagnosis is valid whether or not the preprocessing is enabled. Prints intensity
summaries at three stages of the pipeline — raw, after conversion to 8-bit grey,
after the full transform — and saves a side-by-side panel to
`figures/diag_tb_domain_panel.pdf` and `.png`. The manuscript includes the PDF.
The script takes no arguments.

This is what establishes that the Shenzhen and Montgomery sets differ by about
0.8 normalised intensity units before normalisation and by close to zero after
it, so that the residual cross-site drop is not an artefact of preprocessing.

Produces manuscript Supplementary Fig. S3.

---

## `stats.py`

Library module. The statistical tests behind the comparisons.

| Function | Returns |
|----------|---------|
| `delong_roc_test(y, p_a, p_b)` | `(auc_a, auc_b, p)` for two correlated ROCs |
| `bootstrap_auc_ci(y, p, n_boot=2000)` | `(mean, lo, hi)` |
| `paired_ttest(a, b)` | `(t, p)` across seeds |
| `expected_calibration_error(y, p, n_bins=15)` | ECE |
| `noninferiority_test(ref, ours, margin=0.02)` | `(mean_diff, ci_upper, p, is_noninferior)` |

`noninferiority_test` works on the paired per-seed differences `d = ref - ours`
and tests H0: `mean(d) >= margin` against H1: `mean(d) < margin`.
Non-inferiority is declared when the upper end of the one-sided 95 % interval
for `mean(d)` lies below the margin. A bound above the margin leaves the
question open; it does not establish inferiority.

---

## `models.py`, `datasets.py`, `engine.py`

Library modules.

`models.build_model(name)` accepts `DenseNet121`, `DenseNet121DS`,
`LiteDenseNet`, `DSLiteDenseNet`, `MobileNetV2`, `ShuffleNetV2`. The four
variants come from one parametrised class spanning
{full `[6,12,24,16]` or lite `[4,6,8,6]`} × {standard or depthwise-separable},
so parameter and FLOP counts always derive from the same definitions.

`datasets.CXRDataset` reads a manifest, decodes DICOM or PNG or JPG, applies
CLAHE and the configured normalisation, and returns `(tensor, label)` or, with
`return_meta=True`, the source metadata as well.

`engine.train_model` selects the best epoch by validation AUROC with early
stopping and restores that state before returning. `engine.compute_metrics`
returns the full metric dictionary written to the result CSVs.
`engine.load_imagenet_encoder` initialises the three torchvision baselines from
ImageNet (see `run_ablation.py --imagenet`) and raises unless every encoder
tensor lands; `engine.load_pretrained_encoder` raises if a checkpoint matches no
tensor at all. Neither can train a network from scratch under a pretrained tag
without saying so.

---

## `paper_figures/`

Scripts that draw the schematic and summary figures. They write into `figures/`
by default; pass `--out` to redirect. Output filenames match the names used by
the manuscript's `figs/` directory.

### `dense_block_diagram.py` — Figure 2

Draws the DenseNet-121 and DS-LiteDenseNet dense layers one above the other,
sharing an x grid so the concatenation columns align. Depends on matplotlib
only. Writes `figures/Figure_Dense_Block_Structure.{pdf,png}`.

```bash
python paper_figures/dense_block_diagram.py
```

### `network_architecture.py` — Figure 3

Draws the network end to end: the expanded depthwise-separable convolution and
dense block above, the stem, four dense blocks with transition blocks, and the
classifier below. Depends on matplotlib only. Writes
`figures/Network_Architecture.{pdf,png}`.

```bash
python paper_figures/network_architecture.py
```

### `label_efficiency.py` — Figure 7

Reads `results/ablation_pneumonia_lf<fraction>_scratch_summary.csv` for each of
the five fractions and plots locked-test AUROC against the label budget. It
takes the values from the result files rather than carrying transcribed numbers,
so the figure cannot drift away from the runs behind it. Missing a fraction is
an error naming the command that would produce it. Writes
`figures/label_efficiency.{pdf,png}`.

```bash
# 1.0 is the full-data run of section 3 of the README; repeat it here only if
# that sweep has not been run.
for f in 0.05 0.1 0.25 0.5; do
  python run_ablation.py --task pneumonia --label_fraction $f
done
python paper_figures/label_efficiency.py
```

`--results` points at the directory holding the summary CSVs, `--out` at the
directory the figure is written to, and `--task` selects the task; all three
default to the values used by the manuscript.

### `conv_comparison_standard.py` and `conv_comparison_separable.py` — Figure 1

Draw panel (a), a standard convolution, and panel (b), a depthwise-separable
convolution, as separate files: `figures/Figure_comparison_calculation_a.{pdf,png}`
and `figures/Figure_comparison_calculation_b.{pdf,png}`. The two are combined by
hand into `figs/Figure_comparison_calculation.pdf`.

```bash
python paper_figures/conv_comparison_standard.py
python paper_figures/conv_comparison_separable.py
```

Neither panel carries the parameter-count formulas. They were on the drawings in
an earlier version and duplicated the caption, so they were removed.

### Figures assembled by hand

Two items in the manuscript are laid out in a vector editor rather than produced
by a script:

**Figure 1** is the two panels above placed side by side in one file.

**The graphical abstract** is drawn by hand. Its calibration curve is not
freehand: `export_reliability_curve.py` writes
`reliability_curve_DSLiteDenseNet_pneumonia.csv`, which holds the exact bin
coordinates of the measured curve, and the drawing follows those coordinates.
Every other number on it is quoted from Tables 2, 3, 5 and 6.

### Fonts

Every figure script sets three rcParams before drawing:

```python
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
```

Matplotlib otherwise writes Type 3 fonts into PDF, which are bitmap-based and
are not among the types publishers accept, and draws in DejaVu Sans, which is
not on the usual approved list either. The `fonttype` setting changes only how
glyphs are embedded, not how they look; the family setting does change
appearance, so figures produced before and after it differ slightly.

The family list falls back silently: on a machine without Arial or Helvetica the
figures render in DejaVu Sans and nothing fails. To confirm which font was
actually used, run `pdffonts` on the output and read the name column. If Arial
is installed but not picked up, matplotlib is likely serving a stale font cache;
delete the directory reported by `matplotlib.get_cachedir()` and rerun.
