# Contact: training and evaluation

FAH's contact outputs come from two models on top of each frozen FAH network (`<weights>/<model>/model.pt`). Pose, mesh and
part outputs never change:

- **Image head.** `<weights>/<model>/contact.pt` gives a contact probability for each dense surface point (mapped to the
  778 MANO vertices) and one flag per hand. It is trained with `train.contact_head` and calibrated so that 0.5 is its
  validation threshold.
- **Video model.** `<weights>/<model>/contact_video.pt` is a causal transformer over each hand's track. It is trained with
  `train.contact_video`, and `fah.contact_video` runs it online.

Paths below are placeholders:

| path | contents |
|---|---|
| `$DATA_ROOT` | raw datasets and crop pools (`docs/training.md`) |
| `$DATA_ROOT/contact` | the contact data root: labels `<pool>/`, test sets `test/`, video GT `video_gt/` ([contact_data.md](contact_data.md)) |
| `$WEIGHTS` | the released weights (`<weights>/<model>/`, `<model>` = `fah-s`, `fah-b`, `fah` (FAH-L), `fah-h`) |
| `$CKPT_ROOT` | your checkpoints |
| `$TRACKS` | `bench.run --out_root`: tracks as `$TRACKS/<dataset>/<run_name>/<clip>.pkl` |
| `$RUNS` | your evaluation outputs |

`MANO_DIR` is set as in the README. Labelling needs `open3d`, `trimesh`, `igl` and `scipy`.

## 1. Labels

A MANO vertex is in contact when its signed distance to the nearest GT object surface of the frame is below the tolerance.
The distance is the exact point-to-triangle distance, negative inside an object (winding number), so penetration counts as
contact. Only objects count: not the table, the scene or the other hand.

The head trains on 5 / 10 / 20 / 35 mm, and 10 mm is the main output. InterHand2.6M crops have no objects:
`--zero_sources interhand_lab` labels every vertex 0, and they need no label files.

```bash
# training pools (HOT3D, DexYCB, HO3D, OakInk2 ego / exo): $DATA_ROOT/contact/<pool>/sdf.npy + contact.jsonl
for p in hot3d_lab_mesh dexycb_lab ho3d_lab oakink2_ego_lab oakink2_lab; do
  python -m train.contact_labels --pool $p --crops $DATA_ROOT/crops --data_root $DATA_ROOT --out $DATA_ROOT/contact
done
# ARCTIC (egocentric), HOI4D and MOW: their builders write pools, labels, test sets and clips into the same root
# (contact_data.md); link their crop pools next to the others
ln -s $DATA_ROOT/contact/pools/{arctic_ego,hoi4d,mow_train_c35} $DATA_ROOT/crops/
# held-out test sets (crops + labels) and their per-frame GT (temporal categories, visibility)
for s in hot3d_bench dexycb_s0test; do
  python -m train.contact_labels --test $s --data_root $DATA_ROOT --out $DATA_ROOT/contact
  python -m train.contact_labels --frames $s --data_root $DATA_ROOT --out $DATA_ROOT/contact
done
# the training root of the released heads: links + HOI4D without Chair and without GT penetration deeper than 15 mm
python -m train.contact_labels --make_root $DATA_ROOT/contact/root --out $DATA_ROOT/contact
```

## 2. Image head

Configure [local.json](../configs/README.md), including `model_checkpoint` when using a new pose model.
All sizes use a frozen pose network and a four-layer contact head with a four-block encoder branch.

```bash
python scripts/train.py contact
```

**Calibration.** Run the head on the pools' validation crops and pick the F1-optimal vertex threshold for each tolerance
(pools weighted equally, on the rows HACO also ran on). Then store `contact_bias = logit(threshold)`, which writes
`contact_cal.pt`, the shipped `contact.pt`.

```bash
python -m bench.contact_eval predict --set poolval --model $WEIGHTS/fah/model.pt --ckpt $CKPT_ROOT/contact/fah/contact.pt --tag fah
for k in 0 1 2 3; do python -m bench.contact_eval pick_thr --preds fah --chan $k; done
python -m bench.contact_eval calibrate --ckpt $CKPT_ROOT/contact/fah/contact.pt --thrs 0.72,0.68,0.68,0.58
```

The released heads' thresholds (5 / 10 / 20 / 35 mm) are:

| model | thresholds |
|---|---|
| FAH-S | 0.78 / 0.76 / 0.70 / 0.58 |
| FAH-B | 0.76 / 0.74 / 0.70 / 0.60 |
| FAH-L | 0.72 / 0.68 / 0.68 / 0.58 |
| FAH-H | 0.72 / 0.68 / 0.66 / 0.60 |

## 3. Video model (2-fold stacking)

The video model trains on tracks of heads that never saw those sequences:

1. Train two fold heads with the image-head arguments plus `--steps 8000 --fold_out 0` (or `1`)
   `--fold_pools hot3d_lab_mesh,dexycb_lab,arctic_ego,hoi4d --fold_key sequence`. Calibrate them as above.
2. Run fold head k with `bench.run` on the training clips of fold k. The arguments are the image configuration without the mesh fit,
   and the pickles keep `contact_dense` and `token`.
3. Label the GT of those pickles, then split, train, pick and apply.

```bash
python scripts/train.py contact-folds
python scripts/train.py contact-video
```

Between these commands, calibrate each fold head and generate its predictions only on recordings
excluded from that head's training fold. Set `contact_tracks` to the resulting labelled tracks.
`train.contact_video folds` creates recording-level splits; `bench.run` generates the predictions,
and `train.contact_labels --video_gt` attaches GT. Do not train the video model on in-fold predictions.

FAH-S's video model also trains on the HOI4D and ARCTIC video tracks:
`--train ...,hoi4d:$TRACKS/hoi4d/train-fold,arctic:$TRACKS/arctic/train-fold --exclude hoi4d=_C20_`, which drops HOI4D Chair.

Use the released detector for track generation; rerun scores may differ from the published results.

## 4. Evaluation

The image rows of the contact table use the held-out sets with GT boxes:

- training-disjoint HOT3D (`hot3d_bench`, 5,172 hands);
- DexYCB s0 test (`dexycb_s0test`, 9,506 hands).

The metrics are pooled vertex F1, AP (tied scores grouped) and hand-flag F1, with each model at its validation threshold.
The released heads are calibrated (0.5 = that threshold), so their predictions are scored at 0.5. `predict` loads the
network like every FAH run (`--checkpoint`: the SAM 3D Body checkpoint, README; `--detector_weights`); `<model>` as above.

```bash
# released head
python -m bench.contact_eval predict --set hot3d_bench --model $WEIGHTS/fah/model.pt --ckpt $WEIGHTS/fah/contact.pt --tag fah \
  --detector_weights $WEIGHTS/shared/detector.pt
python -m bench.contact_eval score --set hot3d_bench --preds fah --thr 0.5 --name fah
# HACO (bench/methods/haco.py, in HACO's environment, + hand_tracking_toolkit for HOT3D): the GT box x 2 (HACO's demo crop)
python -m bench.contact_eval haco --set hot3d_bench --data_root $DATA_ROOT
python -m bench.contact_eval score --set hot3d_bench --preds haco_haco_hand2x --thr 0.56 --name haco
# paired sequence-bootstrap CIs (1,000 resamples, seed 2), FAH-L - HACO
python -m bench.contact_ci gt --set hot3d_bench --arms haco=haco_haco_hand2x@0.56,fah=fah@0.5 --out $RUNS/ci_hot3d.json
```

A head you trained yourself (`$CKPT_ROOT/contact/<model>/contact.pt`, before `calibrate`) is scored the way the calibrated
head computes it: `--cal_thr <its 10 mm threshold>` (and `@cal<threshold>` arms in `bench.contact_ci`) applies the bias
before the 400 → 778 map and scores at 0.5:

```bash
python -m bench.contact_eval predict --set hot3d_bench --model $WEIGHTS/fah/model.pt --ckpt $CKPT_ROOT/contact/fah/contact.pt --tag mine \
  --detector_weights $WEIGHTS/shared/detector.pt
python -m bench.contact_eval score --set hot3d_bench --preds mine --thr 0.5 --cal_thr 0.68 --vidx $WEIGHTS/fah/model.pt --name mine
```

HACO's 0.56 is its pool-val threshold, picked like ours: `haco --set poolval --ref <tag>` on the rows of a `predict --set
poolval` file, then `pick_thr --preds haco_haco_hand2x --ref <tag>`.

The full-pipeline rows run `bench.run` with the released head on the test clips (the fold-track arguments of section 3 with
`"contact_ckpt": "$WEIGHTS/fah/contact.pt"`, `--run_name test` -> `$TRACKS/<dataset>/test`), then the released video model
(`train.contact_video apply`), then `bench.contact_eval video`:

- `s1c` is the per-frame row: the shipped image output at 0.5.
- `s4` is the video row: the blend plus the FAH-Video flag (thresholds and hysteresis stored in `contact_video.pt`).

The video model reads the image head's uncalibrated dense logits. `--bias` / `--debias` is the head's 10 mm calibration
bias, logit(t) of its 10 mm threshold t (`contact.pt` `contact_calibration["thresholds"][1]`), and `--thr` is t itself:

| model | t (10 mm) | `--bias` / `--debias` |
|---|---|---|
| FAH-S | 0.76 | 1.1526795 |
| FAH-B | 0.74 | 1.0459686 |
| FAH-L | 0.68 | 0.7537718 |
| FAH-H | 0.68 | 0.7537718 |

```bash
mkdir -p $RUNS/fah
python -m train.contact_video apply --ckpt $WEIGHTS/fah/contact_video.pt --pred_dir $TRACKS/hot3d/test --bias 0.7537718 \
  --out $RUNS/fah/apply_hot3d.npz
python -m bench.contact_eval video --set hot3d_bench --dataset hot3d --pred_dir $TRACKS/hot3d/test --video_npz $RUNS/fah/apply_hot3d.npz \
  --video_ckpt $WEIGHTS/fah/contact_video.pt --vidx $WEIGHTS/fah/model.pt --thr 0.68 --thr_cal 0.5 --debias 0.7537718 \
  --out $RUNS/fah/video_hot3d.json
python -m bench.contact_ci video --set hoi4d_test --dataset hoi4d --subset nochair --arms fah=$RUNS/fah --out $RUNS/ci_hoi4d.json
```

A video model you trained yourself: `--ckpt` / `--video_ckpt $CKPT_ROOT/contact_video/<model>/model.pt` after `pick`.

### Expected numbers (the paper's contact family table)

The GT-box rows come from `predict` + `score --thr 0.5` with each released head above (HACO: `haco` + `score --thr
0.56`). For full-pipeline evaluation, generate tracks with the released detector and video model (section 3).

| method | HOT3D F1 | AP | flag | DexYCB s0 F1 | AP | flag |
|---|---|---|---|---|---|---|
| HACO | 0.58 | 0.47 | 0.82 | 0.66 | 0.61 | 0.86 |
| FAH-S | 0.77 | 0.83 | 0.98 | 0.75 | 0.81 | 0.98 |
| FAH-B | 0.79 | 0.85 | 0.98 | 0.76 | 0.82 | 0.98 |
| FAH-L | 0.80 | 0.85 | 0.98 | 0.76 | 0.82 | 0.98 |
| FAH-H | 0.80 | 0.86 | 0.98 | 0.77 | 0.82 | 0.98 |
| FAH-L, per frame (full pipeline) | 0.80 | 0.85 | 0.98 | 0.76 | 0.81 | 0.98 |
| FAH-L + FAH-Video (full pipeline) | 0.80 | 0.86 | 0.97 | 0.77 | 0.81 | 0.97 |

The other tables in the supplement use the same commands:

- HOI4D (`hoi4d_test`, without Chair: `bench.contact_ci --subset nochair`), ARCTIC (`arctic_val`) and MOW (predictions from
  `train.build_mow_contact preds`, then `mow_score` or `scripts/contact_mow_pairs.py`), with test sets from their builders
  ([contact_data.md](contact_data.md));
- the rule rows: `bench.contact_eval geometric`, the predicted mesh with the GT object meshes and the 10 mm rule;
- flag switches and onset / release / hover from `video`, with CIs from `bench.contact_ci`.

## 5. Figures

- The main contact figure (`scripts/contact_main_fig.py`): per hand input | GT | HACO | FAH-L on the GT mesh, six hands
  picked by a fixed rule (seeded order, grasp slots per dataset, one FAH-L failure slot) from the GT-box test predictions
  above.
- The supplement's contact panels (`scripts/contact_qual.py`): the released FAH-L on the paper's picks,
  `scripts/contact_qual_picks.json` (HOT3D held-out clips + COCO photos, with the photo credits).

```bash
python scripts/contact_main_fig.py select --fah fah@0.5 --haco haco_haco_hand2x@0.56 --out $RUNS/fig/sel.json
python scripts/contact_main_fig.py render --sel $RUNS/fig/sel.json --cap_px 45 --out $RUNS/fig
python scripts/contact_qual.py render --picks scripts/contact_qual_picks.json --hot3d $DATA_ROOT/hot3d/train_aria \
  --coco_root $DATA_ROOT/coco --out $RUNS/panels
```
