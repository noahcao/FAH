# Third-party notices

This repository contains code adapted from the projects below, depends on packages that you install yourself, and
downloads some files at run time. Each keeps its own licence. This repository's own code is licensed under the Apache
License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)); the files derived from SAM 3D Body and DINOv3 keep their licences (below);
all model weights, including the hand detector, use [WEIGHTS_LICENSE](WEIGHTS_LICENSE).

## Code included in this repository

| component | licence | where | notice |
|---|---|---|---|
| [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) (Meta) | SAM License | the 25 files listed below | file headers (copyright + `SPDX-License-Identifier: LicenseRef-SAM-License`); [SAM-License.txt](third_party_licenses/SAM-License.txt) |
| [DINOv3](https://github.com/facebookresearch/dinov3) (Meta) | DINOv3 License | `fah/models/backbones/dinov3_vit/` (vendored, 11 files) | file headers (`SPDX-License-Identifier: LicenseRef-DINOv3-License`); [DINOv3-License.md](third_party_licenses/DINOv3-License.md) (also `fah/models/backbones/dinov3_vit/LICENSE.md`). Built with DINOv3. |
| [TRAM](https://github.com/yufu-wang/tram) | MIT | `bench/world_slam.py` (the masked DROID-SLAM stream, mask resizing, tracking loop and scale fit) | licence text in the file; [TRAM-MIT.txt](third_party_licenses/TRAM-MIT.txt) |
| [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) | BSD 3-Clause | `bench/world_slam.py` (through TRAM: DROID-SLAM's demo arguments and stream) | [DROID-SLAM-BSD-3-Clause.txt](third_party_licenses/DROID-SLAM-BSD-3-Clause.txt) |
| [DPVO](https://github.com/princeton-vl/DPVO) | MIT | `bench/world_dpvo.py` (`masked_patchify`, adapted from `Patchifier.forward`) | [DPVO-MIT.txt](third_party_licenses/DPVO-MIT.txt) |
| [torchvision](https://github.com/pytorch/vision) | BSD 3-Clause | `train/camera_motion.py` (`graph_safe_raft`, adapted from RAFT's correlation block) | [torchvision-BSD-3-Clause.txt](third_party_licenses/torchvision-BSD-3-Clause.txt) |
| [RoMa](https://github.com/naver/roma) | BSD 3-Clause | `fah/models/modules/sync_free.py` (`rotmat_to_unitquat`, `unitquat_to_euler`) | [RoMa-BSD-3-Clause.txt](third_party_licenses/RoMa-BSD-3-Clause.txt) |
| [Metric3D](https://github.com/YvanYin/Metric3D) | BSD 2-Clause | `bench/world_slam.py` (`Metric3DDepth`, its inference path) | [Metric3D-BSD-2-Clause.txt](third_party_licenses/Metric3D-BSD-2-Clause.txt) |
| [pymomentum](https://github.com/facebookresearch/momentum) (Meta) | MIT | `fah/models/heads/fk_eager.py`, `fah/models/heads/mhr_hand_fast.py` (eager ports of the MHR rig's forward kinematics and pose features) | [pymomentum-MIT.txt](third_party_licenses/pymomentum-MIT.txt) |
| [MHR](https://github.com/facebookresearch/MHR) (Meta) | Apache License 2.0 | the same two files (ported from the MHR TorchScript model) | [MHR-Apache-2.0.txt](third_party_licenses/MHR-Apache-2.0.txt) |

Files under the SAM License (from SAM 3D Body, modified for FAH): `fah/config.py`, `fah/data/prepare_batch.py`, `fah/data/transforms/__init__.py`, `fah/data/transforms/bbox_utils.py`, `fah/data/transforms/common.py`, `fah/hand_model.py`, `fah/models/backbones/__init__.py`, `fah/models/backbones/dinov3.py`, `fah/models/backbones/vit.py`, `fah/models/decoders/__init__.py`, `fah/models/decoders/prompt_encoder.py`, `fah/models/decoders/promptable_decoder.py`, `fah/models/heads/__init__.py`, `fah/models/heads/camera_head.py`, `fah/models/heads/mhr_head.py`, `fah/models/modules/__init__.py`, `fah/models/modules/camera_embed.py`, `fah/models/modules/drop_path.py`, `fah/models/modules/geometry_utils.py`, `fah/models/modules/layer_scale.py`, `fah/models/modules/mhr_utils.py`, `fah/models/modules/misc.py`, `fah/models/modules/swiglu_ffn.py`, `fah/models/modules/transformer.py`, `fah/models/optim/fp16_utils.py`.

Files under the DINOv3 License (vendored from DINOv3): `fah/models/backbones/dinov3_vit/__init__.py`, `fah/models/backbones/dinov3_vit/layers/__init__.py`, `fah/models/backbones/dinov3_vit/layers/attention.py`, `fah/models/backbones/dinov3_vit/layers/block.py`, `fah/models/backbones/dinov3_vit/layers/ffn_layers.py`, `fah/models/backbones/dinov3_vit/layers/layer_scale.py`, `fah/models/backbones/dinov3_vit/layers/patch_embed.py`, `fah/models/backbones/dinov3_vit/layers/rms_norm.py`, `fah/models/backbones/dinov3_vit/layers/rope_position_encoding.py`, `fah/models/backbones/dinov3_vit/utils.py`, `fah/models/backbones/dinov3_vit/vision_transformer.py`.

`fah/estimator.py`, `fah/metadata.py` and `fah/models/heads/mask_head.py` reuse a few interface-level details of SAM 3D
Body (the batch key names, the MHR70 keypoint indices, the default intrinsics, a LayerNorm2d layer as in Segment
Anything); the rest of the repository is FAH's own code.

The SAM 3D Body files keep upstream comments on their own origins (timm, DINOv2, DETR, Segment Anything, HMR 2.0 / HaMeR,
RotationContinuity). `fah/_vendor/` holds our own minimal stand-ins (an mmcv subset re-exporting mmengine for Metric3D,
a torch_scatter shim for DPVO), not copies of those packages. `fah/fast_mano.py` implements the MANO model itself and
`fah/models/heads/mano_head.py` its own forward kinematics (on smplx's MANO layer); the MANO model files are not
included. `fah/yolo.py` implements the YOLOv8 detection / segmentation network itself and reads the detector checkpoints
without the Ultralytics package.

## Packages you install

Not included in this repository; installed with `pip install -e .` (requirements.txt) or an extra, each under its own
licence:

- core: PyTorch and torchvision (BSD 3-Clause), NumPy (BSD 3-Clause), OpenCV (Apache 2.0), RoMa (BSD 3-Clause), einops
  (MIT), yacs (Apache 2.0), OmegaConf (BSD 3-Clause), huggingface_hub (Apache 2.0), SciPy (BSD 3-Clause), Pillow
  (MIT-CMU), timm (Apache 2.0), dill (BSD 3-Clause);
- **Ultralytics** (AGPL-3.0): not needed for inference (`fah/yolo.py` reads the detector checkpoint and runs the network
  without it); only the `train` extra (detector training) and the `eval` extra (the baselines' detectors) install it;
- **smplx** (MPI licence for non-commercial scientific research): the MANO layer used for MANO faces and the
  benchmarks (`bench/common.py`);
- extras: TensorRT (`trt`, NVIDIA licence), MoGe (`world`, MIT), mmengine and pytorch-minimize (`offline`), ONNX and
  NVIDIA Model Optimizer (`engine`), coremltools (`coreml`, BSD 3-Clause), Weights & Biases (`train`), the HOT3D hand
  tracking toolkit, mcap, protobuf, gdown and PyYAML (`eval`), Open3D (MIT), libigl (MPL 2.0) and trimesh (MIT)
  (`contact`: the contact label builders);
- separate environments (docs/world.md): DPVO (MIT), TRAM's DROID-SLAM (BSD 3-Clause), Metric3D (BSD 2-Clause).
- real to sim (docs/real2sim.md): MuJoCo (Apache 2.0), CoACD (MIT), trimesh (MIT), imageio and imageio-ffmpeg (BSD
  2-Clause; FFmpeg builds under LGPL / GPL); in their own environments SAM 3 and SAM 3D Objects (SAM License, weights
  gated on Hugging Face) and MoGe-2 (MIT); optional dex-retargeting (MIT) and the dex-urdf robot descriptions (each robot
  under its own licence).

The baseline adapters in `bench/methods/` call the official code of WiLoR, HaMeR, Hamba, HaWoR, AnyHand, MediaPipe Hands,
RTMPose and HACO from their own checkouts or packages; none of their code is included, and each is used under its own
licence (several are non-commercial). The same holds for the segmentation baselines (`bench/seg/baselines.py`,
fetched by `scripts/setup_seg_baselines.sh`): SAM 2.1 (Apache 2.0), SAM 3 (SAM License; weights gated on Hugging Face),
detectron2 (Apache 2.0), VISOR-HOS and Hands23 (their repositories' terms), and for the SAM 3D Body hand baseline
(`bench/methods/sam3db.py`: the SAM 3D Body checkpoint, SAM License, downloaded by the user).

## Files downloaded at run time

- SAM 3D Body `model_config.yaml`: [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3),
  SAM License (gated; `fah.download_sam3d`).
- `assets/mhr_model.pt`: the MHR model from the [MHR release v1.0.1](https://github.com/facebookresearch/MHR/releases/tag/v1.0.1),
  Apache License 2.0 (`fah.download_sam3d`; the same file as in the SAM 3D Body repository).
- MANO: [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de), MANO licence (non-commercial research; registration).
- Dyn-HaMR (the supplement's world-space comparison; `scripts/dynhamr/setup*.sh` fetch it at a pinned commit, nothing
  is included): MIT License, with its README limiting the code and model to non-commercial research use. Its setup
  installs chumpy (MIT), configer and human_body_prior (MPI licence for non-commercial scientific research), the MANO
  package (MPI MANO licence, non-commercial), Hand-BMC-pytorch (MIT; its joint statistics come from RHD and GANerated
  Hands under their terms), hand_tracking_toolkit (Apache 2.0) and, for the paper-time version, detectron2 and mmcv
  (Apache 2.0), the HaMeR fork (MIT) and ViTPose (Apache 2.0); the MANO, HaMeR and HMP weights under their own terms.
- World mode: DPVO's `dpvo.pth`, MoGe-2 (`Ruicheng/moge-2-vitl-normal`, MIT), and for the offline camera DROID-SLAM's
  `droid.pth` and the Metric3D weights, each from its project under its own terms.
- FAH provides the hand-detector weights (`shared/detector.pt`) under [WEIGHTS_LICENSE](WEIGHTS_LICENSE).
  They use the [Ultralytics](https://github.com/ultralytics/ultralytics) YOLOv8 architecture and trainer;
  applicable Ultralytics terms remain in effect. Inference uses FAH's own implementation.
- The camera-motion head (`shared/camera_motion.pt`) contains torchvision's pretrained RAFT-small weights (trained on
  FlyingChairs and FlyingThings3D, whose terms allow research use only).

No dataset is redistributed; the builders read each dataset from its official source under its own licence
([docs/training.md](docs/training.md), [docs/evaluation.md](docs/evaluation.md)).
