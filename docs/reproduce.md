# Evaluating the release

Use the released FAH checkpoints to evaluate images, videos, contact and world-space tracking.
See the [paper](https://arxiv.org/abs/2611.07711) for the reported results. This guide covers the
supported release; reruns can differ from the published numbers. Dataset images, stored predictions,
and checkpoints for paper-only experiments are not bundled.

## Setup

Follow [Installation](../README.md#install), install the `eval` extra, and set `DATA_ROOT`, `RESULTS`,
`FAH_WEIGHTS` and `MANO_DIR`. Build TensorRT engines for GPU timing as described in
[Acceleration](acceleration.md). Dataset downloads, splits and metric definitions are in
[Evaluation](evaluation.md).

## 3D pose and meshes

```bash
python -m bench.run --dataset hot3d --split bench/splits/hot3d_bench.json \
  --data_root $DATA_ROOT/hot3d --method fah \
  --method_args '{"model":"fah","mode":"image"}' --run_name fahL-image --out_root $RESULTS
python -m bench.evaluate $RESULTS/hot3d/fahL-image --mano_dir $MANO_DIR
```

Use `"mode":"video"` for causal refinement and a different run name. Model names are `fah-s`,
`fah-b`, `fah` (FAH-L) and `fah-h`. For DexYCB or EgoStandard, use their dataset names, roots and
splits from [Evaluation](evaluation.md#3d-benchmarks). Keep EgoStandard frames and predictions private;
report aggregate metrics only.

## 2D keypoints

```bash
python -m bench.run --dataset bench2d --data_root $DATA_ROOT/bench2d_v2 \
  --method fah --method_args '{"model":"fah","mode":"image"}' \
  --run_name fahL-image --out_root $RESULTS
python -m bench.bench2d.evaluate $RESULTS/bench2d/fahL-image
```

Build the benchmark and synthetic low-light variant with the commands in
[Evaluation](evaluation.md#2d-benchmarks). Some source images cannot be redistributed;
only evaluate sets whose images and annotations you have obtained.

## Contact, parts and world space

- [Contact](contact.md#4-evaluation): calibrated image/video contact, HACO comparison and confidence intervals.
- [Part segmentation](evaluation.md#part-segmentation-and-dense-points): segmentation and dense-point metrics.
- [World space](world.md#evaluation-and-timing): causal/offline cameras, trajectory metrics and timing.
- [Real to sim](real2sim.md): reconstruction, simulation and scoring.

## Figures and timing

- `scripts/render_teaser.py`: the five model outputs on input images.
- `scripts/paper_fig2.py`: pipeline illustration tiles.
- `scripts/compare_figure.py`: SAM 3D Body and FAH comparison.
- `scripts/contact_main_fig.py` and `scripts/contact_qual.py`: contact comparisons and examples.
- `scripts/supp_samples.py`: custom-set examples.
- `scripts/benchmark_frames.py` and `scripts/benchmark_latency.py`: end-to-end and per-stage timing.

Each script documents its arguments in `--help`. Preserve source credits when exporting figures.
For training your own checkpoints, use [Training](training.md) and [Contact training](contact.md).
