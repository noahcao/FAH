# Training

The recipes live in [scripts/train.py](../scripts/train.py). Configure paths and your own resource
settings once, then run the stage you need. Dataset preparation is in [Training data](training_data.md).
Fresh training runs, especially with public data only, may differ from the published scores.

## Configure once

```bash
cp configs/local.example.json configs/local.json
```

Edit `configs/local.json`: set `data_root`, `ckpt_root`, `weights`, `mano_dir`, `sam_checkpoint`,
`train_processes`, `train_batch_size` and `train_devices` for your setup. The local file is Git-ignored.
`sam_checkpoint` is the full SAM 3D Body checkpoint, needed for training only.
See [Configuration](../configs/README.md) for optional settings.

## Pose model

After preparing the crop pools, run these stages in order:

```bash
python scripts/train.py teacher
python scripts/train.py distill
python scripts/train.py heads
python scripts/train.py pose
```

The default model is FAH-L (`fah`). Use `--model fah-s`, `fah-b` or `fah-h` for another size.
Set the model in your config, or pass the same `--model` to `distill`, `heads` and `pose`.
Each model initialises its own heads directly on its distilled encoder; no extra ViT-B run is needed.

`pose` is one 30k-step fine-tune with a single cosine learning-rate schedule. It uses available
external KD targets throughout and writes `<ckpt_root>/finetune/<model>/pose/final.pt`.
Only the teacher is shared across sizes. Pose training resumes both the encoder and heads from
`<ckpt_root>/finetune/<model>/heads/final.pt`. This simplified recipe has not been
validated against the published checkpoints.

## Optional components

| Component | Command | Prepared input |
|---|---|---|
| Part head | `python scripts/train.py parts` | Pose checkpoint and part labels |
| Image contact | `python scripts/train.py contact` | Pose checkpoint and contact labels |
| Fold contact heads | `python scripts/train.py contact-folds` | Same labels; used to generate out-of-fold tracks |
| Video contact | `python scripts/train.py contact-video` | Out-of-fold contact tracks |
| Detector masks | `python scripts/train.py detector-labels` | Full-frame boxes/keypoints; SAM 3 environment |
| Detector | `python scripts/train.py detector` | The generated detector labels and released detector weights |
| MANO fitter | `python scripts/train.py mesh` | Extracted prediction/GT pairs (`mesh_data`) |
| MHR-Hand fitter | `python scripts/train.py mhr` | Extracted prediction/GT pairs (`mhr_data`) |
| Pose video refiner | `python scripts/train.py video` | Per-frame pose tracks (`pose_tracks`) |
| Camera-motion head | `python scripts/train.py camera` | Frame caches from `train.build_cam_frames` |

Parts follow the new pose checkpoint automatically. Contact stages use the released pose checkpoint
unless `model_checkpoint` points to your newly trained one. See [Contact](contact.md) for label
preparation, fold separation and calibration. The camera recipe writes its benchmark-exclusion list
automatically; it is a training template, not an exact reconstruction of the published checkpoint.

The detector script generates SAM 3 labels directly, then low-light copies; it does not train an
intermediate SAM 2.1 detector. Set `sam_python` if SAM 3 uses a separate environment. Rejected masks
use a keypoint-expanded ellipse: inspect `qc_summary.json` and the labels before detector training.

## Preview or change settings

```bash
python scripts/train.py pose --dry-run
```

`--dry-run` prints the complete commands without starting training or writing generated files.
Use `--config path/to/config.json` for another configuration. Recipe losses and schedules are
centralised in the script; paths, pool mixtures and resource choices belong in your local config.
