# Evaluation

All benchmarks run in two steps: `bench.run` writes per-clip prediction pickles (GT + predictions per frame), and an
evaluator turns a run directory into `metrics.json`. FAH models are passed as `--method fah`:

```bash
export MANO_DIR=$PWD/data/mano DATA_ROOT=$PWD/data RESULTS=$PWD/output/bench
FAH='{"model": "fah", "mode": "image"}'   # mode: image | video; + "weights": DIR, "trt_dir": "<weights>/trt" (TensorRT)
python -m bench.run --dataset hot3d --split bench/splits/hot3d_bench.json --data_root $DATA_ROOT/hot3d \
    --method fah --method_args "$FAH" --run_name fah-image --out_root $RESULTS [--shard i --num_shards n]
python -m bench.evaluate $RESULTS/hot3d/fah-image --mano_dir $MANO_DIR
```

`--method fah_pipeline` takes the full pipeline arguments instead of a model and mode (`fah.method_args(model, mode)`
prints them). `bench/run.py` also has adapters for the baselines of the paper tables (WiLoR, HaMeR, HaWoR, Hamba, the AnyHand
weights, MediaPipe Hands, RTMPose); see "Baselines" below. No dataset is redistributed: each builder reads
the official release, `bench/splits/` only lists the clips / episodes of our splits.

## 3D benchmarks

Metrics (`bench.evaluate`): PA-MPJPE / PA-MPVPE (mm, Procrustes-aligned joints / MANO vertices), RR-MPJPE
(root-relative), on hands matched by box IoU >= 0.3, and the coverage-aware `*_p` variants that charge every missed GT
hand the error of a canonical MANO hand (so detection misses count). Video runs (`mode: video`) also report
acceleration error and jitter.

| benchmark | data | split | build |
|---|---|---|---|
| HOT3D | HOT3D-Clips (Aria), the 27 validation sequences of HaWoR's split of HOT3D's train_aria recordings (disjoint from our training) | `bench/splits/hot3d_bench.json` (`_full`: all clips of these sequences) | download the listed clips (below) |
| DexYCB | DexYCB, toolkit setup s0 test, one camera per sequence | `bench/splits/dexycb_s0_test.json` (`_full`: all 8 cameras) | `python -m bench.datasets.build_dexycb --raw $DATA_ROOT/dexycb_raw --out $DATA_ROOT/dexycb --download` |
| EgoStandard | LightwheelAI/EgoStandard, 120 x 30 s clips of held-out tasks (test only) | `bench/splits/egostandard_bench.json` | `python -m bench.datasets.build_egostandard --raw $DATA_ROOT/egostandard_raw --out $DATA_ROOT/egostandard` (needs `HF_TOKEN` with access to the dataset), then `python -m bench.datasets.egostandard_fit --root $DATA_ROOT/egostandard --mano_dir $MANO_DIR` |

HOT3D clips (Hugging Face dataset `bop-benchmark/hot3d`; needs `pip install -e ".[eval]"` for the HOT3D toolkit):

```python
import json, os
from huggingface_hub import hf_hub_download
split = json.load(open("bench/splits/hot3d_bench.json"))
for dev, clips in split["clips"].items():
    for c in clips:
        hf_hub_download("bop-benchmark/hot3d", c, repo_type="dataset", local_dir=os.path.join(os.environ["DATA_ROOT"], "hot3d"))
```

DexYCB and EgoStandard runs use `--dataset dexycb --data_root $DATA_ROOT/dexycb` (paths in the split are
relative to it) and `--dataset egostandard --data_root $DATA_ROOT/egostandard`. EgoStandard's licence allows
evaluation but not redistribution of frames or records: keep its frames, crops and overlays private and publish
aggregate metrics only. `bench/splits/egostandard_dev.json` is a disjoint dev split (other tasks, same converter:
`--split bench/splits/egostandard_dev.json --task_rem 1 --n_clips 80`) used only to select the world camera's scale
settings; train on neither split.

## 2D benchmarks

`python -m bench.run --dataset bench2d --data_root <set> ...` then `python -m bench.bench2d.evaluate $RESULTS/bench2d/<run>`:
PCK@0.05 / 0.1 / 0.2 of the joints normalised by the GT keypoint box (matched hands; `*_all` counts missed hands as
wrong), recall, handedness accuracy, precision on fully annotated subsets.

- **bench2d** (in the wild + egocentric): HOT3D (projected MANO joints), CMU Panoptic hand labels, COCO-WholeBody val
  hands, HInt New Days / EPIC-KITCHENS VISOR:
  `python -m bench.bench2d.build --out $DATA_ROOT/bench2d_v2 --version bench2d_v2 --hot3d_root ... --cmu_root ... --coco_root ... --hint_root ...`
- **low light**: synthetic low light on bench2d (`python -m bench.bench2d.make_dark --src $DATA_ROOT/bench2d_v2
  --dst $DATA_ROOT/bench2d_v2_dark --level dark`), merged with real dark images by `bench.bench2d.merge_sets`; the
  real dark photos are not released (their images may not be redistributed).
- **shadows** and **hard in-the-wild hands**: our labels are in [datasets/shadow_v1](../datasets/shadow_v1/) and
  [datasets/hard_mined_v2](../datasets/hard_mined_v2/); `python datasets/fetch_images.py --set datasets/shadow_v1 --out
  $DATA_ROOT/shadow_v1` downloads the images from their original hosts. The **challenging-pose** set is not released
  (its images may not be redistributed). Any bench2d-format set (`images/` + `annotations.json`) is scored with
  `python -m bench.custom_eval --ann <set>/annotations.json --runs $RESULTS/bench2d/<run>` (the paper's 2D-table cells;
  hands marked `needs_label` are not scored). See [datasets/README.md](../datasets/README.md).
- **hand segmentation** (instance masks; EPIC-KITCHENS VISOR val, or FreiHAND): `bench.seg.build` /
  `bench.seg.build_freihand`, `bench.run --dataset seg`, `bench.seg.evaluate` (`bench.seg.evaluate_hand` cuts the GT at
  the wrist for hand-only masks). The segmentation baselines of the mask table run on the same frames with
  `scripts/seg_compare.py` (see "Baselines").

`python -m bench.summarize --results $RESULTS --out $RESULTS/leaderboard.md` collects every run's metrics.

## Part segmentation and dense points

Both evaluations use held-out labelled crops of the training pools (the "Data mix" pools of
[training.md](training.md), built by `train.build_*_crops`): crops of the training validation split (`train.finetune`
VAL_MOD rule, never trained on) whose reference part label was clipped to a hand mask. The paper's crops are listed in
`bench/splits/parts_heldout.json` (200 per pool; HOT3D, DexYCB, OakInk2-ego, COCO-WholeBody, HInt),
`bench/splits/dense_parts.json` (300 per GT-mesh pool) and `bench/splits/parts_hard.json` (109 difficult HOT3D crops
whose labels passed a visual review); `bench.parts.heldout.sample` is the rule they were drawn with. Reference labels and
masks (0 background, 1 palm, 2-16 finger segments, 255 ignore), as in [training.md](training.md) "Part head":

```bash
for p in hot3d_lab_mesh dexycb_lab oakink2_ego_lab; do   # GT MANO mesh per part, clipped to a SAM 2.1 mask that agrees with it
  python -m train.sam_masks --pool $DATA_ROOT/crops/$p --out_dir sam_parts/masks
  python -m train.part_masks --pool $DATA_ROOT/crops/$p --mask_dir sam_parts/masks --min_mask_iou 0.7 --out_dir masks_parts
done
for p in coco_wb_kin hint_lab_kin; do                         # a trained FAH model's mesh per part inside the pool's hand mask
  python -m train.part_labels_model --pool $DATA_ROOT/crops/$p --out_dir masks_parts_model --method_args '<FAH pipeline args>'
done
```

**Part head** (`scripts/part_seg_eval.py`, `bench.parts.seg`): each crop runs twice, with its label box as the only
detection (part map vs the labels: `pixel_acc`, `miou_parts`) and with the model's own detector (the detected hand that
overlaps the SAM mask most: `detbox_miou_parts` over the 16 parts on detected hands, `n_detected`, the part union vs the
SAM mask `detbox_union_iou_sam` / `detbox_union_bf_sam`; `e2e_*` counts a missed hand as background). The raw head is
scored (`parts_in_mask` false) unless set in `--method_args`.

```bash
export FAH_WEIGHTS=... MANO_DIR=$PWD/data/mano
python scripts/part_seg_eval.py --crops $DATA_ROOT/crops --method fah \
    --method_args '{"model": "fah", "mesh": "none", "tips": false, "parts": true}' \
    --save_preds output/parts/fah --out output/parts/fah.json            # model: fah-s | fah-b | fah | fah-h
python scripts/part_seg_eval.py --crops $DATA_ROOT/crops --from_preds output/parts/fah --out x.json   # metrics only
```

FAH-L (`"model": "fah"`), the paper's part-segmentation table row (`detbox_miou_parts` / `n_detected`; union vs SAM
mask IoU / boundary F):

| pool | COCO (`coco_wb_kin`) | HInt (`hint_lab_kin`) | HOT3D (`hot3d_lab_mesh`) | DexYCB (`dexycb_lab`) | OakInk2 (`oakink2_ego_lab`) |
|---|---|---|---|---|---|
| part mIoU, detected hands | 0.51 (0.5142) | 0.53 (0.5259) | 0.54 (0.5399) | 0.46 (0.4553) | 0.46 (0.4600) |
| detected hands | 184 | 197 | 199 | 200 | 200 |
| union vs SAM mask, IoU / BF | 0.83 / 0.39 | 0.78 / 0.26 | 0.83 / 0.30 | 0.82 / 0.28 | 0.81 / 0.25 |

With the paper's detector TensorRT engine every metric is identical to the paper's run; with the PyTorch detector (no
`trt_dir`) the detected-hand mIoU moves by at most 0.0005 (the same two-decimal values) and the box-mode metrics are
identical. Labels or masks in other directories: `--part_dirs` / `--mask_dirs pool:dir,...`.

**Dense points** (`scripts/dense_part_eval.py`, `bench.parts.dense`): the raw dense surface points of each arm (a model
or checkpoint) on each crop's label box, scored against the labels: `wrong_finger` (visible finger points on another
finger's region), `part_acc_fingers_strict` (on their own segment, points on ignore pixels counted as errors),
`cover_fingers` (label pixels within 0.02 view sides of a visible point of their part), `sil_iou` (discs around the
points vs the labelled hand); the "3d" entry is the mean over the three GT-mesh pools. `--geometry` adds breakdowns by
finger crossings, occlusion and hand size (GT meshes); several arms get paired bootstraps (`--boot`, `--boot_unit clip`).

```bash
python scripts/dense_part_eval.py --crops $DATA_ROOT/crops --arms fah=fah --geometry --out_dir output/dense/fah
python scripts/dense_part_eval.py --crops $DATA_ROOT/crops --arms a=runs/a/final.pt,b=runs/b/final.pt --out_dir output/dense/ab
python scripts/dense_part_eval.py --crops $DATA_ROOT/crops --split bench/splits/parts_hard.json --arms fah=fah --out_dir output/dense/hard
```

FAH-L, "3d" entry: `wrong_finger` 0.065, `part_acc_fingers_strict` 0.52, `cover_fingers` 0.31, `sil_iou` 0.68 (its pose
network is trained without the part terms). Use `--arms name=<checkpoint>` to evaluate your own trained checkpoints.

## World space

`python -m bench.world <video run dirs> --cams gt,<camera dir>` maps camera-frame hands to the world with per-frame
camera poses and scores the true-positive frames of each GT hand with the protocol of
[WHAM](https://github.com/yohanshin/WHAM) (alignment from [SLAHMR](https://github.com/vye16/slahmr)) and the hand
conventions of [HaWoR](https://github.com/ThunderVVV/HaWoR) (`--protocol wham`, the default). A hand's frames are
split at missing frames into runs of at least 16 frames (`--min_run`), and the predicted world joints are aligned to the
GT ones with Sim(3) fits, in two variants:

- **100-frame chunks** (`--chunk 100`, the default; `world_<cams>_wham100.json`): each run is cut into 100-frame chunks
  (the remainder is kept); W-MPJPE100 (mm) is fitted on a chunk's first two frames, WA-MPJPE100 (mm) on the whole
  chunk;
- **whole tracks** (`--chunk 0`; `world_<cams>_wham0.json`): the same with one chunk per run: W-MPJPE on the run's
  first two frames, WA-MPJPE on the whole run;
- RTE (%), in both: per run, a rigid fit of the wrist trajectory; wrist error / GT wrist path length x 100.

All metrics are averaged over frames, both hands included (`*_right` / `*_left`: one side; `*_bridge`: missing frames
skipped instead of split). `--protocol rigid` (one rigid alignment per track) is a legacy option, not used in the
paper. Camera sources: `gt`, `none`, or a directory of `<clip>.npz` with `T_world_cam`; the FAH camera is described in
[world.md](world.md).

## Timing

All speeds are end-to-end per frame (frames decoded into memory first, the GPU synchronised around every frame):

```bash
python scripts/benchmark_latency.py cache --data_root $DATA_ROOT --out /tmp/latency_frames   # 4 EgoStandard + 4 HOT3D streams
python scripts/benchmark_latency.py run --cache /tmp/latency_frames --method fah \
    --method_args '{"model": "fah", "mode": "image", "trt_dir": "<weights>/trt"}' --tag fah --out $RESULTS/latency
python scripts/benchmark_frames.py --frames <dir of <stream>_<index>.jpg> --method fah --method_args '<json>'   # any device (CUDA / MPS / CPU)
python scripts/benchmark_world_e2e.py ...    # world pipeline: hands + camera processes on one GPU (docs/world.md)
```

`benchmark_latency.py` reports the median latency of an uninstrumented pass (fps = 1000 / median ms) and a stage split
from a second, instrumented pass. Detectors alone (the detector and mask-cost tables): `scripts/benchmark_detector.py
--image <frame> --weights a.pt b.pt b.pt a.pt --det_size 800 --warmup 50 --iters 500` (boxes, and boxes + the masks of the
top-2 boxes for YOLO-seg weights; `--half` / `--trt` per file for fp16 / TensorRT; networks the fast path does not
implement, such as WiLoR's YOLOv8-pose detector, are timed through the ultralytics call). Paper speeds: the slower of two
rounds.

## Baselines

The adapters in `bench/methods/` expect the official code in `$THIRDPARTY/<name>` (default `thirdparty/`) with its own
environment and weights: [WiLoR](https://github.com/rolpotamias/WiLoR), [HaMeR](https://github.com/geopavlakos/hamer),
[HaWoR](https://github.com/ThunderVVV/HaWoR), [Hamba](https://github.com/humansensinglab/Hamba),
[AnyHand weights](https://huggingface.co/chen-si-02/AnyHand-Models) (WiLoR / HaMeR checkpoints), MediaPipe Hands
(`hand_landmarker.task`) and RTMPose (rtmlib ONNX models). Each adapter's docstring states the settings used.

**SAM 3D Body hand** (`--method sam3db_hand`, `bench/methods/sam3db.py`): the released SAM 3D Body network (`model.ckpt`
from the gated [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3), SAM License) through the
FAH pipeline code with none of FAH's weights or post-processing: WiLoR's detector (confidence 0.3, 512 px), the hand
decoder's native 512 px crops (box padding 0.9), bf16 with CUDA graphs (`{"cuda_graph": false}`: the eager timing). Pass
`{"checkpoint": "<dir>/model.ckpt"}` (the directory also holds `model_config.yaml` and `assets/mhr_model.pt`).

**Segmentation baselines** (`bench/seg/baselines.py`, `scripts/seg_compare.py`; environments, checkouts and weights at the
paper's versions: `bash scripts/setup_seg_baselines.sh`): SAM 2.1 hiera-B+ / -L and SAM 3 prompted with FAH's hand boxes,
SAM 3 with the text prompt "hand", VISOR-HOS and Hands23 with their own detections (thresholds 0.5 / 0.7), on the frames
of a `bench.run --dataset seg` run of FAH with `"det_masks": true`; each arm's run is scored like FAH's
(`bench.seg.evaluate`, and `bench.seg.evaluate_hand --key mask_rle --margin 0 --cut_ref <FAH run> --ref_key mask_det`
for one hand-only GT cut shared by every method). Speeds: the arm's call on the decoded frame (prompt boxes precomputed,
so the detector is excluded), fps = 1000 / median of the per-chunk median frame times, slower of two runs.
The contact baseline is [HACO](https://github.com/dqj5182/HACO_RELEASE) (`bench/methods/haco.py`, `$THIRDPARTY/HACO`, its
ViT-H checkpoint, CC BY-NC 4.0): hand box x 2 crops at threshold 0.56 on HOT3D / DexYCB, its mask box x 1.3 crops at 0.52 on
MOW; the contact data (HOI4D, ARCTIC, MOW) and its builders: [contact_data.md](contact_data.md).
