# bench2d_v2: public 2D hand keypoints

6,364 images with 8,582 hands. All labels are the upstream datasets' own, so this folder ships no labels. It only has
`fingerprints.json`: per image id, its subset, an MD5 of its labels and the first 16 hex digits of the SHA-256 of its
image file.

| subset | images / hands | source | labels |
|---|---|---|---|
| `hot3d` | 1,459 / 2,570 | HOT3D-Clips (Aria), every 10th frame of the 98 clips of `bench/splits/hot3d_bench.json` | projected MANO joints, fisheye rectified to an upright pinhole camera; every hand labelled (precision scored) |
| `cmu` | 461 / 846 | CMU Panoptic hand dataset, `hand_labels/manual_test` (MPII + NZSL images) | manual 2D |
| `coco` | 836 / 1,558 | COCO-WholeBody v1.0 val on COCO val2017 | manual 2D; hands with >= 8 labelled joints and a box >= 32 px |
| `hint_nd`, `hint_epick` | 1,723 / 1,723, 1,885 / 1,885 | HInt test: New Days, EPIC-KITCHENS VISOR (the Ego4D part is excluded) | manual 2D (joints with existence = 1) |

## Build

Download the upstream data first:

- HOT3D-Clips: the clips listed in `bench/splits/hot3d_bench.json`; see docs/evaluation.md.
- CMU Panoptic hand dataset `hand_labels.zip`: http://domedb.perception.cs.cmu.edu/handdb.html
  (http://domedb.perception.cs.cmu.edu/panopticDB/hands/hand_labels.zip)
- COCO-WholeBody `coco_wholebody_val_v1.0.json` (https://github.com/jin-s13/COCO-WholeBody) and COCO `val2017`.
- HInt `HInt_annotation_partial.zip`: https://fouheylab.eecs.umich.edu/~dandans/projects/hamer/HInt_annotation_partial.zip

Then build and check:

```bash
python -m bench.bench2d.build --out $DATA_ROOT/bench2d_v2 --version bench2d_v2 --mano_dir $MANO_DIR \
    --hot3d_root $DATA_ROOT/hot3d --cmu_root <dir with hand_labels/> --coco_root <dir with coco_wholebody_val_v1.0.json and val2017/> \
    --hint_root <HInt_annotation_partial dir with TEST_newdays_img/ and TEST_epick_img/>
python datasets/check_set.py --ann $DATA_ROOT/bench2d_v2/annotations.json --ref datasets/bench2d_v2/fingerprints.json --images
```

On 2026-10-30, a rebuild of the `hot3d`, `cmu` and `coco` subsets with this repository's builder matched the paper's
copy in every label and every image byte. We did not re-check the `hint_*` subsets, because the HInt download was not
at hand.

FAH-L on this set: PCK@0.1 0.83, incl. misses 0.81, 3D fit 0.83, overlapping hands 0.64
(`python -m bench.custom_eval --ann $DATA_ROOT/bench2d_v2/annotations.json --runs <run> --fit3d --overlap`).

## Licence

The images and labels follow the upstream terms. HOT3D's hand annotations are CC BY-NC-SA. CMU Panoptic is for
research. The COCO-WholeBody annotations are for research and non-commercial use only. The COCO images are Flickr photos
under their own licences. HInt's annotations are MIT. EPIC-KITCHENS is CC BY-NC 4.0. `fingerprints.json` contains no
labels.
