# Parametric hands: MANO and MHR-Hand

FAH's network predicts 21 joints, a dense MANO-topology surface and their 2D positions. A parametric hand is fitted to
these after the network, in one of two models, or both:

- **MANO** (`--mesh mano`, the default): the MANO mesh stage, a learned one-pass fitter plus one Gauss-Newton step.
- **MHR-Hand** (`--mesh mhr`, alias `hmr`, or `--mesh both`): the hand of the Momentum Human Rig (MHR) as SAM 3D Body's
  hand decoder uses it ([fah/mhr_fit.py](../fah/mhr_fit.py)).
  - Fitted: the 27 right-hand joint angles, SAM 3D Body's hand scale and shape components, and the hand's rotation and
    translation in the camera frame.
  - Steps: a closed-form per-part initialisation, a learned one-pass initialiser (`shared/mhr_fitter.pt`), then one
    damped Gauss-Newton step with the analytic rig Jacobian, placed onto the 2D joints (PnP, as the MANO stage). All of
    it runs as one CUDA graph, and it is deterministic: the same input gives the same output on every run.
  - Use it when you need MHR / SAM 3D Body parameters, for example to drive the full-body MHR rig or to compare with
    SAM 3D Body.

```bash
fah --img hand.jpg --mesh both --output_dir output/fah     # MANO and MHR-Hand meshes: overlays (MANO | MHR-Hand) + JSON
```

```python
import fah
hands = fah.load("fah", mesh="both")(image_bgr, K)
hands[0]["verts3d_mhr"], hands[0]["mhr_params"]["model_params"]
```

**Standalone:** fit any MANO-topology surface (778 points) + 21 joints:

```python
import fah
from fah.mhr_fit import MHRHandFitter, fit_mhr_hand

net = fah.load("fah").method.est.model                    # any FAH model: its model.pt carries the MHR head buffers
fitter = MHRHandFitter.from_model(net, device="cuda", iters=1, init_net=fah.mhr_fitter())
hand = fit_mhr_hand(fitter, points3d, joints3d, is_right=True, K=K, joints2d=joints2d)   # (778, 3), (21, 3) or batches
```

Inputs:
- `points3d` (778, 3): a MANO-topology hand surface.
- `joints3d` (21, 3): the 21 joints in OpenPose order (wrist, thumb 1-4, index 1-4, middle 1-4, ring 1-4, pinky 1-4).
- Both are in the camera frame, in metres (FAH's `verts3d_dense` / `joints3d`).
- `is_right`: the hand's side.
- Optional `K` (3, 3) with `joints2d` (21, 2): places the fitted hand rigidly so that its joints project onto the 2D
  joints (PnP).

## Outputs

Per hand, in FAH's hand dicts with `mesh="mhr"` / `"both"` (`fit_mhr_hand` keys in brackets):

| key | shape | content |
|---|---|---|
| `verts3d_mhr` (`verts`) | (2445, 3) | MHR hand mesh, camera frame, metres; faces `model.faces_mhr` (right; reversed winding for left hands) (`faces`) |
| `verts3d_mhr_mano` (`verts_mano`) | (778, 3) | the MHR mesh mapped to MANO topology (`fah/assets/mhr_to_mano778.npz`) |
| `joints3d_mhr_fit` (`joints`) | (21, 3) | MANO-convention kinematic joints of the MHR mesh (MHR -> MANO21 regressor) |
| `joints3d_mhr_rig` (`joints_rig`) | (21, 3) | the rig's own skeleton joints at SAM 3D Body's 21 hand keypoints (MHR convention: other wrist / thumb / tip locations than MANO) |
| `mhr_params` (`params`) | dict | below |
| `verts3d_mhr_world` | (2445, 3) | `--mode world`: the MHR hand mesh in the world frame |

`mhr_params` holds SAM 3D Body's and MHR's parameters of the fitted hand:
- `hand_pose` (27): the right hand's joint angles (MHR model parameters, radians, SAM 3D Body's joint order).
- `hand_pca` (54): SAM 3D Body's hand-decoder code. It is the continuous code (6D per 3-DoF joint, sin / cos per angle)
  minus its mean; the hand components are the identity.
- `scale` (28): SAM 3D Body's scale components. Only 8 (wrist scale) and 18-27 (bone lengths) are fitted; the others
  only move the body.
- `shape` (45): identity components. Only 40-44 move the hand and are fitted.
- `global_rot` (3): SAM 3D Body's wrist-centric global rotation (euler xyz).
- `global_rotmat` (3, 3): the hand's camera-frame rotation.
- `cam_t` (3): SAM 3D Body's camera translation.
- `model_params` (204): the MHR rig's input. `mhr(shape, model_params, zeros(72))[0] / 100`, with y and z negated, plus
  `cam_t`, is the full-body MHR mesh whose right hand is `verts3d_mhr`.
- `mirrored`: true for left hands. As SAM 3D Body does, a left hand is the right MHR hand fitted to the mirrored hand.
  Mirror x of the rig's output to get it back; `verts3d_mhr` is already mirrored.

## Accuracy

mm; FAH image model; every GT-matched hand of the benchmarks (HOT3D 25,388, DexYCB 9,240); both meshes
fitted to the same network output and placed by PnP:

| HOT3D / DexYCB | PA-MPVPE | PA-MPJPE |
|---|---|---|
| dense points (input) | 5.64 / 6.60 | 4.31 / 5.40 |
| MANO fit | 3.96 / 5.12 | 4.17 / 5.30 |
| MHR-Hand fit | 4.51 / 5.44 | 4.48 / 5.37 |

- The MHR mesh is scored through its MANO-topology map, and joints are each fit's MANO-convention joints. MHR's rig
  keypoints follow MHR's own convention and are not comparable to MANO ground truth.
- The MHR hand cannot represent MANO hands exactly: fitted to the ground-truth MANO meshes themselves (a converged
  fit), it stays 2.23 / 1.99 mm away; the shipped MANO stage on ground-truth inputs: 1.03 / 1.45.
- `scripts/mhr_fit_eval.py` computes this table from stored `bench.run` predictions.

## Speed

FAH-L image model, desktop RTX 4070 Ti SUPER (Windows 11 + WSL2), TensorRT, fps = 1000 / median ms on HOT3D frames,
the slower of two rounds: `--mesh mano` (default) 65 fps, `--mesh mhr` 65, `--mesh both` 59. `scripts/benchmark_mhr_fit.py`
times the fit alone on the CPU.

## Files

- `assets/mhr_model.pt`, the MHR model, from Meta's MHR release v1.0.1 (the same file SAM 3D Body ships), already part
  of `fah.download_sam3d()`. From SAM 3D Body: the hand-branch buffers of MHR's head
  (scale / shape components, the hand code's mean, the hand-mode wrist transform, the keypoint mapping) are in every
  FAH `model.pt`; the fit's rig is built from them when the model loads.
- From FAH:
  - `shared/mhr_fitter.pt`: the learned initialiser (`train/mhr_fitter.py`). Without it the fit starts from its
    closed-form initialisation and takes 2 steps.
  - `fah/assets/mhr_to_mano778.npz` (`scripts/fit_mhr_mano_regressor.py`, fitted on training splits only) and
    `fah/assets/mhr_to_mano21_kin_regressor.npz`: the MHR -> MANO correspondences.

## Licence

The MHR model (`mhr_model.pt`: skeleton, skinning, blend shapes, pose correctives) comes from Meta's
[MHR](https://github.com/facebookresearch/MHR) release v1.0.1 under the Apache License 2.0 (byte-identical to the copy
in SAM 3D Body). The hand parameterisation comes from SAM 3D Body
([facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3)) and is used under the SAM License,
which you accept when downloading its model config from the gated repository. The MHR-Hand fitter's outputs are
produced with these. The MHR -> MANO correspondences and the learned initialiser are FAH files derived from
fits of the MHR model.
