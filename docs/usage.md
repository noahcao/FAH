# Usage and outputs

[Installation and model selection](../README.md) · [Acceleration](acceleration.md)

## Quick start

```bash
cp configs/local.example.json configs/local.json  # edit paths and output options
python scripts/infer.py hand.jpg
python scripts/infer.py clip.mp4
python scripts/infer.py clip.mp4 --mode world
```

Use `--model fah-s` for FAH-S, or change `model` in the config. `--dry-run` previews the command.
Backend, optional outputs and camera-environment paths are configured in
[local.json](../configs/README.md). The `fah` CLI below remains available for individual options.

Options (`fah --help`): `--backbone s|b|l|h` (FAH-S / FAH-B / FAH-L, the default / FAH-H), `--mesh none|mano|mhr|both`
(MANO by default; `mhr`, alias `hmr`: the MHR-Hand mesh; `both`: both, drawn side by side), `--dense --parts --masks`
(save and draw these outputs), `--mode image|video|world` (for `--vid`; video by default), `--device` / `--dtype`
(bfloat16 on CUDA and Apple MPS by default), `--trt` / `--coreml`, `--focal` / `--fov`, `--weights`, `--no_tips` (no
fingertip step), `--min_score` (overlay threshold only). World mode runs the camera model
([world.md](world.md): DPVO and MoGe-2 in their own environments, `--dpvo_dir` / `--dpvo_python` /
`--depth_python`; FAH ships a minimal torch_scatter shim for DPVO) and writes the camera trajectory
(`<name>_cameras.npz`) and world-frame joints; `--cameras` uses a trajectory you already have. `--camera_version
offline` (default `realtime`) adds an offline camera for recorded videos when accuracy matters more than speed (not real time):
masked DROID-SLAM (TRAM's mask-aware version) with a Metric3D metric scale, its scale averaged with the real-time
camera's, and the real-time camera where the two disagree (e.g. a near-static head); it needs DROID-SLAM and Metric3D in
their own environment ([world.md](world.md#offline-camera-not-real-time)). `python demo.py ...` is the same
command without the console script.

Python:

```python
import cv2, fah

model = fah.load("fah", mode="image")                             # mode: image | video | world; mesh="mano" | "none" | "mhr" | "both"
hands = model(cv2.imread("hand.jpg"), K=None)                     # K: 3x3 intrinsics (default: focal = image diagonal)
for h in hands:
    h["is_right"], h["joints3d"], h["joints2d"], h["verts3d"]       # (21, 3) m, (21, 2) px, (778, 3) MANO mesh
```

For videos call `model.reset()` before each clip, then `model(frame, K)` frame by frame. Joints follow the OpenPose
order (wrist, thumb 1-4, index 1-4, middle 1-4, ring 1-4, pinky 1-4) with the MANO joint definition; 3D outputs are in
the camera frame (x right, y down, z forward, metres). `fah.method_args(model, mode)` returns the full configuration
of a model and mode.

## Run on your own video

```bash
fah-video my_video.mp4 -o output/my_video.mp4 --trt   # FAH-L, every output: output/my_video.{mp4,pkl} + _poster.jpg
```

One command runs the video model (its video detector, the MANO fit, part maps, instance masks, the contact
head and the video contact model) frame by frame and writes every output per frame to `<out>.pkl` (format in
`fah-video --help`), then renders all of them: input | 2D joints | MANO mesh, and part maps + instance masks | per-vertex
contact | palm-side contact per hand, over the per-hand contact timeline (H.264 without audio; needs `ffmpeg`).
`--model fah-s|fah-b|fah|fah-h`, `--panels input,joints,mesh,parts,contact,palm` (any subset, 3 per row),
`--no-render` (results only), `--load <out>.pkl` (render saved results again), `--world realtime|offline` (+ a world
panel with the camera model or the offline camera, in their environments: [world.md](world.md)),
`--min-score` (display threshold, 0.3), `--device cuda|mps|cpu`, `--focal` / `--fov` / `--K fx,fy,cx,cy` (default:
focal = image diagonal), `--trt` / `--trt_dir` (TensorRT engines from `fah-build-trt`), `--no_names` (generic panel
titles and legend, without model names). The project page's in-the-wild test videos
([project page][page-videos]) are this command's output: `fah-video
self_test_video.mov --fov 70 --no_names --trt` and `fah-video ego_test_video_real.mp4 --world offline --no_names --trt`.

## Outputs

Per hand (Python API keys; the `fah` JSON holds the same values, unrounded):

| output | API key | CLI JSON |
|---|---|---|
| side, confidence, box (xyxy px) | `is_right`, `score`, `bbox` | always |
| 21 joints, 2D (px) and 3D (camera frame, m) | `joints2d`, `joints3d` | always |
| fitted MANO mesh (778 vertices) | `verts3d` (`mesh="mano"` / `"both"`) | `--mesh mano` (default) / `both` |
| fitted MHR-Hand mesh (2,445 vertices, faces `model.faces_mhr`), its MANO-topology map (778), its MANO-convention and rig joints (21) | `verts3d_mhr`, `verts3d_mhr_mano`, `joints3d_mhr_fit`, `joints3d_mhr_rig` (`mesh="mhr"` / `"both"`) | `--mesh mhr` / `both` |
| MHR-Hand parameters (SAM 3D Body's, [mhr_hand.md](mhr_hand.md)) | `mhr_params` | `--mesh mhr` / `both` |
| dense points: 40 bone + dense surface points, 2D / 3D / visibility; the dense surface as a MANO-topology mesh | `surface2d_head`, `surface3d_head`, `surface_visible`, `verts3d_dense` | `--dense` |
| part map (0 background, 1 palm, 2-16 finger segments) inside the instance mask | `parts_roi` = (x0, y0, map) (`parts=True`) | `--parts` (RLE per part; `fah.cli.decode_parts`) |
| instance mask (full image, RLE) | `mask_det` (with `det_masks=True`) | `--masks` |
| hand-object contact: per-vertex probability on the 778 MANO-topology vertices, on the dense points, per hand, per palm / finger; at 5 / 10 / 20 / 35 mm | `contact`, `contact_dense`, `contact_score`, `contact_fingers`; `contact_k`, `contact_score_k` (`contact=True`) | `--contact` |
| video contact (FAH, videos): per-vertex probability (778), per hand, hand in contact | `contact_video`, `contact_score_video`, `contact_flag_video` (`contact=True` in video / world mode; `contact_video=True` for the frames of a video in image mode) | `--vid ... --contact` |
| world-frame joints / mesh, camera pose | `joints3d_world`, `verts3d_world`, `verts3d_mhr_world` (`fah.world.place`) | `--mode world` (+ `T_world_cam` per frame) |

`--dense`, `--parts`, `--masks` and `--contact` also draw these outputs in the overlays. Instance masks (and the part maps
inside them) are split between two hands that share a detector box (crossed or praying hands) by the hands' mesh
silhouettes.

### Contact

Enable `contact=True` / `--contact`. The frozen pose outputs stay unchanged. `contact` is the calibrated
10 mm per-vertex probability; `contact_k` contains 5 / 10 / 20 / 35 mm outputs. Vertex and finger contact
use >= 0.5; the hand output uses > 0.5. `contact_fingers` is ordered palm, thumb, index, middle, ring, pinky.
Labels cover object contact; predictions can also fire on unlabelled hand-hand or body contact.
Core ML exports omit contact: use PyTorch on Mac.

Video/world mode adds `contact_video`, `contact_score_video` and the hysteresis flag `contact_flag_video`.
For ordered frames in image mode, set `contact_video=True` and reset before each clip.
Frames filled by the pose refiner have no contact output. `contact_smooth=2` / `--contact_smooth 2`
reduces flag switches at about one frame of latency and a small flag-F1 cost; only the flag changes.
The default is 1. See [contact training and evaluation](contact.md) for architecture, data and calibration.

**MANO or MHR-Hand** ([mhr_hand.md](mhr_hand.md)). Both fit the same predicted points.
Use `mesh="mano"`, `"mhr"` or `"both"`; `mhr_params` drives SAM 3D Body's MHR hand rig.
On HOT3D, FAH-L runs at 65 fps with either fit and 59 fps with both.
See [the fit comparison](mhr_hand.md#accuracy) for accuracy and representation limits.

[page-videos]: https://noahcao.github.io/FAH/#test-videos
