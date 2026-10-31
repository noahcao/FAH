# Evaluation sets

The 2D sets of the paper that we made, or that are built from public datasets with our scripts. We ship our labels,
a per-image list of sources and licences, and scripts that download or rebuild the images. We do not redistribute any
images.

| set | images / scored hands | in this folder | images from | our labels |
|---|---|---|---|---|
| [shadow_v1](shadow_v1/) | 166 / 274 | `annotations.json`, `images.jsonl` | HaGRID v2, Open Images V7, Openverse, Wikimedia Commons (original hosts, `fetch_images.py`) | yes |
| [hard_mined_v2](hard_mined_v2/) | 167 / 169 | `annotations.json`, `images.jsonl` | HaGRID v2, Open Images V7 (original hosts, `fetch_images.py`) | yes |
| [bench2d_v2](bench2d_v2/) | 6,364 / 8,582 | `fingerprints.json` | HOT3D-Clips, CMU Panoptic, COCO-WholeBody val, HInt test (`bench.bench2d.build`) | no (upstream labels) |
| [low_light](low_light/) | 6,364 / 8,582 | `fingerprints.json` | synthetic low-light copy of bench2d_v2 (`bench.bench2d.make_dark`) | no (bench2d_v2 labels) |

Not released: the challenging-pose set (hard_pose_v1) and the 19 real photos of the low-light set (dark_v1). Their
images are stock photos and web images that we may not redistribute. Four images of the paper's 170-image shadow_v1
are left out for the same reason.

## Get the images, run, score

Prerequisite for the Open Images photos: download `downloader.py` from https://github.com/openimages/dataset and install
its listed requirements.

`fetch_images.py` writes the Open Images ids of a set to a list and runs the downloader on it (`--oi_downloader`;
`--oi_python` if its requirements live in another environment). Without it, the photos come from their Flickr copies.
Some of these are not the bytes the set used, and the report lists them.

```bash
python datasets/fetch_images.py --set datasets/shadow_v1 --out $DATA_ROOT/shadow_v1 --oi_downloader downloader.py   # also hard_mined_v2
python -m bench.run --dataset bench2d --data_root $DATA_ROOT/shadow_v1 \
    --method fah --method_args '{"model": "fah", "mode": "image"}' --run_name shadow-fah-l --out_root $RESULTS
python -m bench.custom_eval --ann $DATA_ROOT/shadow_v1/annotations.json --runs $RESULTS/bench2d/shadow-fah-l
```

`fetch_images.py` checks every file against its SHA-256 (`exact`). It also accepts a file with the same decoded
pixels (`same_pixels`; some Flickr copies come back with their metadata stripped) or, failing that, the same image size
(`size_only`, listed in `fetch_report.json`). On 2026-10-30, with the Open Images downloader, all 333 images came back
`exact` (327) or `same_pixels` (6).
bench2d_v2 and low_light are built from the upstream datasets (see their READMEs); `check_set.py` compares a local
build with the copy the paper used.

`bench.custom_eval` scores the prediction files of a `bench.run` run against an annotation file and prints the paper's 2D-table cells. These
are recall, PCK@0.1 of the hand size on matched hands, and PCK@0.1 with missed hands counted as wrong ("incl."), plus
PCK@0.05, fingertip PCK@0.1 and handedness. Add `--fit3d --overlap` for the bench2d_v2 3D-fit and overlapping-hands
columns. Hands are matched by mean visible-keypoint error at most 0.5 of the hand size. Hands marked `needs_label` are
not scored, and predictions on them are not false positives. Precision only counts images marked `fully_annotated`.

## Reproduction check (FAH-L, expected values)

`bench.custom_eval` on the paper's FAH-L predictions (the prediction files are not released; their scores are). The "paper copy" column reproduces the
published cells exactly. The "released copy" column is what a run on the released or rebuilt sets should give. Its raw
values are under `expected` in `expected_fah_l.json`.

| set | cells | paper copy | released copy |
|---|---|---|---|
| bench2d_v2 | PCK / incl. / 3D fit / overlapping hands | 0.83 / 0.81 / 0.83 / 0.64 | the same (identical set) |
| hard_mined_v2 | PCK@0.1 | 0.98 | the same (identical set) |
| shadow_v1 | recall / PCK / incl. | 1.00 / 0.99 / 0.99 (170 images) | 1.00 / 0.99 / 0.99 (166 images) |
| low light | recall / PCK / incl. | 0.96 / 0.81 / 0.78 (19 real + synthetic) | 0.97 / 0.81 / 0.78 (synthetic only) |

## Licences

Images keep their own licences, listed per image in `images.jsonl`. When you use or show them, credit the author and
licence given there. Our annotation files (`shadow_v1/annotations.json`, `hard_mined_v2/annotations.json`) are
covered by [LICENSE.md](LICENSE.md). The bench2d_v2 and low-light labels are the upstream datasets' and follow their terms.
