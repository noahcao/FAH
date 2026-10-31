# Training data

Set `data_root` in `configs/local.json`. Crop pools live in `<data_root>/crops`.
The launchers use the public pose pools below, sampled by weight rather than dataset size.
Override `pose_pools` or `contact_pools` in your config with `[name, weight]` pairs.
Private hard-example and HaGRID pseudo-label pools are excluded from the default recipe.
Fresh public-data runs can differ from published results.

| pool | weight | data (labels) | builder |
|---|---|---|---|
| hot3d_lab_mesh | 1.3 | HOT3D-Clips train, non-benchmark sequences (MANO) | `train.build_hot3d_crops --labels --verts --stride 2 --name hot3d_lab_mesh` |
| coco_wb_kin | 1.0 | COCO-WholeBody train2017 hands (2D) | `train.build_crops --source coco_wb`, then `train.convert_2d_labels` |
| cmu_lab_kin | 0.3 | CMU Panoptic hand labels, manual_train (2D) | `train.build_crops --source cmu`, `train.convert_2d_labels` |
| hint_lab_kin | 0.6 | HInt train: New Days + EPIC-KITCHENS (2D) | `train.build_crops --source hint`, `train.convert_2d_labels` |
| freihand_lab_mesh_kin | 0.5 | FreiHAND train (MANO) | `train.build_crops --source freihand`, `train.add_freihand_verts`, `train.freihand_kin convert` |
| egodex_test_mano | 0.6 | EgoDex test (ARKit 3D, converted to the MANO joint convention) | `train.build_egodex_crops`, `train.qc_crops`, `train.convert_egodex_labels --corr train/assets/arkit_to_mano_wilor_mesh_kin_pa.json` |
| egodex_part1_mano | 0.6 | EgoDex part 1 (as above) | as above |
| visor_lab | 0.8 | EPIC-KITCHENS VISOR train (GT masks) | `train.build_visor_crops --detector --name visor_lab` |
| neg | 0.5 | background crops of exhaustively labelled frames (VISOR, HOT3D) | `train.build_neg_crops --det <detection dataset>` |
| halpe_lab_kin | 0.8 | Halpe-FullBody / HICO-DET (2D) | `train.build_halpe_crops`, `train.convert_2d_labels` |
| dexycb_lab | 0.5 | DexYCB s0 train (MANO) | `train.build_dexycb_crops --labels --verts --stride 8` |
| interhand_lab | 0.5 | InterHand2.6M 5 fps, NeuralAnnot (MANO) | `train.build_interhand_crops --verts` |
| oakink2_lab | 0.4 | OakInk2, all views (MANO) | `train.build_oakink2_crops --stride 8 --verts` |
| oakink2_ego_lab | 0.5 | OakInk2 egocentric view (MANO) | `train.fetch_oakink2_view`, `train.build_oakink2_crops --views egocentric --stride 4 --min_joints_in 15 --per_subject 20000 --verts --name oakink2_ego_lab` |
| ho3d_lab | 0.3 | HO-3D v3 (MANO) | `train.build_ho3d_crops --stride 3 --verts` |
| hanco_lab | 0.2 | HanCo, validated fits (MANO) | `train.build_hanco_crops --verts` |
| hard_lab | 0.3 | hard 2D examples (HaGRID / Open Images), reviewed | not released (our data engine) |
| hagrid_pseudo | 0.5 | HaGRID consensus pseudo-2D | not released (our data engine) |

EgoDex crops are filtered by a label check against a trained model (`train.qc_crops`; `train.finetune` keeps crops
with a 2D disagreement < 0.12 of the hand size from `qc_*.jsonl`; the released lists: `train.fetch_qc`, see "Notes")
and SAM masks by `train.qc_masks` (keypoints inside
the mask >= 0.5, mesh IoU >= 0.2). `_kin` pools hold the same images with the 2D labels moved to the kinematic MANO
joint convention (`train/assets/label2d_offsets.json`); `--verts` pools store the GT MANO vertices; SAM masks
(`train.sam_masks`) are added per pool. Rename or link each builder's output directory to the pool name used in
`--sources`. Each builder's docstring lists its options; check every dataset's licence before use (several are
research-only).

## Teacher inputs

The teacher fine-tune uses `hot3d_lab`, `coco_wb`, `cmu_lab`, `hint_lab`, `freihand_lab`,
`egodex_test` and `halpe_lab`. Encoder distillation uses `hot3d_lab_mesh`, `coco_wb` and `cmu_lab`.
Build these base pools before starting the corresponding stage.

## Masks and part labels

Use `train.sam_masks` to prepare crop masks. The pose/part recipe uses SAM 2.1; SAM 3 is supported
with `--model sam3` but requires validation of the resulting labels. This is independent of the
SAM 3 detector-label recipe.

For GT-mesh part pools (HOT3D, DexYCB, OakInk2-ego, HanCo and FreiHAND), write crop masks to
`sam_parts/masks`, outside the pose-supervision `masks` directory. Build labels with `train.part_masks`
using that mask directory and `--min_mask_iou 0.7`. On COCO, HInt and CMU, use
`train.part_labels_model` with `--out_dir masks_parts_model` and the trained pose model.

## External targets and QC

Build WiLoR targets with `train.wilor_kd` where required. Keep them in the pools:
the pose launcher uses available targets throughout its single training run. No files need to be moved.

Use the supplied label-convention files in `train/assets/`. The released QC ID lists are distributed
with the weights; `train.fetch_qc` checks their hashes and installs them into the crop pools.

Exclude benchmark sequences from all training data: HOT3D's HaWoR validation sequences, DexYCB s0
test, and both EgoStandard evaluation splits. Inspect camera `data_inventory.json` before evaluation.
