# World space

FAH (world) = the video model's hands (`mode: world`) placed in a world frame with a causal metric camera (the reported
world numbers and timings ran with the mesh stage off, `--mesh none`; the world joints do not depend on it):

1. **DPVO** (causal visual odometry, every 2nd frame at 384 px, hand boxes of frame t-1 masked, tracking-failure
   re-initialisation) with a **piecewise metric scale** from 15 MoGe-2 depth calls, one every 30 fed frames, each step
   scaled by the running median of the DPVO instance's calls so far (`bench/world_dpvo.py --scale_rmed`,
   `bench/world_depth.py`), corrected by the factor fitted on HOT3D-train clips (x1.034, `bench.world_calib fit`), and a
   translation gate on steps that DPVO explains by rotation alone;
2. the **learned camera-motion head** (`train/camera_motion.py`, `bench/world_camhead.py`): metric relative poses with
   predicted errors from consecutive fed frames;
3. **causal fusion** (`bench/world_fuse.py`): DPVO's steps with the scale corrected by the head over the last 30 fed
   steps (geometric mean of the two metric scales, `--w_head 0.5`), the head's step wherever DPVO's rotation disagrees by
   more than 3x the head's predicted error or DPVO is not tracking yet (`--k_sigma 3 --bridge`). The head's correction
   fades out with parallax: its scale exponent goes from 0.5 to 0 as DPVO's translation evidence (rotation-only minus
   full-pose residual, median over the last 30 fed steps) rises from 0.5 to 1.5 px (`--ev_lo 0.5 --ev_hi 1.5`), where
   metric visual odometry's translation is trusted. A step keeps DPVO's rotation but takes the head's translation where
   the two translations differ by more than 10x the head's predicted translation error (`--t_sigma 10`), and by more
   than 3x on steps where the hand masks leave less than 70 % of DPVO's patch grid free on either fed frame (`--t_sigma_low
   3 --t_sigma_cov 0.7`; the static coverage `sig_cov` of `bench/world_dpvo.py`): where the hands fill the frame, DPVO's
   few patches sit on the hand edges and on moving objects, and its steps jump.

These are the camera model's settings (`fah.world.DPVO_ARGS` / `FUSE_ARGS`, `fah --mode world`, the default
`--camera_version realtime`). DPV-SLAM's loop closure was tested and not adopted.

Then `fah --mode world --cameras <fused>/<clip>.npz` (or `fah.world.place`) maps each frame's camera-frame joints /
mesh with `T_world_cam`. Any other metric camera trajectory in the same format (`T_world_cam (N, 4, 4)` camera -> world,
optional `index (N,)`) can be used instead.

## Requirements

- [DPVO](https://github.com/princeton-vl/DPVO) built in its own environment (its README; `dpvo.pth` in the checkout),
  checked out at `$THIRDPARTY/DPVO` (tested with torch 2.11 cu128, numpy < 2). FAH ships a minimal torch_scatter shim for DPVO
  (`fah/_vendor/torch_scatter.py`), always first on DPVO's path: it reproduces the reported world numbers exactly.
  Installing PyG's torch-scatter in the DPVO environment also works (slightly different numerics; take the shim's
  directory off the path to use it).
- [MoGe-2](https://github.com/microsoft/MoGe) (`Ruicheng/moge-2-vitl-normal`, or a local `model.pt` via `MOGE_CKPT`):
  `pip install -e ".[world]"` puts it in the FAH environment (the default `--depth_python`); or point `--depth_python` at
  its own environment. DPVO reaches it through a subprocess.
- The camera-motion head: `<weights>/shared/camera_motion.pt` (`fah.camera_head(weights)`; self-contained, no
  download at inference), or one trained with `train/camera_motion.py` on `train/build_cam_frames.py` caches
  (docs/training.md).

## One command

```bash
python scripts/infer.py clip.mp4 --mode world
```

Set the camera paths and interpreters in `world` in [local.json](../configs/README.md).
The script runs the causal camera pipeline and saves overlays, predictions and camera trajectories.
Set `world.cameras` to use an existing trajectory. Keep the same intrinsics for camera and hand inference.

## Offline camera (not real time)

For recorded videos where speed does not matter, `--camera_version offline` (`fah.world.estimate_cameras_offline`)
adds a SLAM trajectory to the real-time camera:

1. the real-time camera above;
2. masked DROID-SLAM over every frame, with the hands' fitted MANO meshes masked out, and a per-clip metric scale from
   Metric3D (ViT-L) depth on its keyframes: the scale of the SLAM depth to the metric depth outside the hands in a
   0.4-0.7 m band (widened while undefined), median over keyframes (`bench/world_slam.py`);
3. `fah.world.fuse_offline`: the SLAM trajectory (rotations and shape) with its scale moved halfway, in log scale, to the
   real-time camera's (the similarity scale between the two trajectories); where the two disagree (mean similarity
   residual above 5% of the real-time camera's path length, e.g. a near-static head where SLAM fails) the real-time
   camera is used.

It needs the hands' meshes (`--mesh mano` or `both`, the default) and its own environment: the mask-aware DROID-SLAM
of [TRAM](https://github.com/yufu-wang/tram) (`thirdparty/DROID-SLAM` there, built as in DROID-SLAM's README;
`droid.pth` from [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)), a
[Metric3D](https://github.com/YvanYin/Metric3D) checkout with its ViT-L weights (`metric_depth_vit_large_800k.pth`),
and `pip install -e ".[offline]"` (mmengine, pytorch-minimize, timm) in that environment. FAH's
`fah/_vendor` (torch_scatter and mmcv stand-ins) goes first on its path. The weights are downloaded under their
projects' terms.

The paper's offline and SLAM rows ran the masked DROID-SLAM + Metric3D steps from HaWoR's checkout, which is CC BY-NC-ND
4.0 and therefore not used here. `bench/world_slam.py` runs the same steps with TRAM's mask-aware DROID-SLAM (MIT; its
DROID-SLAM sources are the same as HaWoR's copy apart from the autocast API) and an upstream Metric3D checkout. On 3
HOT3D benchmark clips (clip-001881, clip-002490, clip-003034) and 1 EgoStandard dev clip, with the same FAH-L hand masks
and the real-time camera of seed 0, the offline camera's W-MPJPE100 / whole-track W-MPJPE (mm, mean ± SD over runs):

| clips | this repository (3 runs) | HaWoR's checkout |
|---|---|---|
| 3 HOT3D | 26.41 ± 0.02 / 26.46 ± 0.03 | 26.49 ± 0.13 / 26.57 ± 0.15 (4 runs) |
| 1 EgoStandard dev | 23.06 ± 0.29 / 30.44 ± 1.40 | 23.06 ± 0.29 / 30.32 ± 1.00 (3 runs) |

The difference stays within the run-to-run spread of either source; the camera rotations differ by 0.02 degrees (HOT3D)
and 0.18 degrees (EgoStandard) between the sources, against 0.01 / 0.22 degrees between runs of HaWoR's checkout.

```bash
python scripts/infer.py clip.mp4 --mode world
```

Set `world.camera_version` to `"offline"` and supply the SLAM/Metric3D paths in your local config.

The output has the same format and conventions as the real-time camera (`clip_cameras.npz`, `T_world_cam` per frame;
`<work>/cams_offline/<clip>.npz` also stores `k_ours_over_slam` and `fallback`). It does not run in real time.
For benchmark runs whose two cameras already exist (a `bench/world_slam.py` run with the FAH hands as `--mask_run`, and
a real-time camera run from `bench/world_fuse.py`), the same fusion runs clip by clip:
```bash
python -m bench.world_offline --slam <slam dir> --ours <camera dir> --out <offline dir>
```
Use the real-time camera for live or long videos and when speed matters; the offline camera when accuracy matters more
than speed and the head moves enough for SLAM.

Known issue: run-to-run variation of the camera. Two runs with the same seed do not give bit-identical cameras: DPVO's
bundle-adjustment kernel and the scatter aggregation of its update network add floating-point values with GPU atomics,
whose order changes between runs, and on long, low-parallax clips DPVO's keyframe and re-initialisation decisions amplify
these last-bit differences. Over 5 repeats with a fixed seed, the world scale varies by 0.1 % (SD) on the 5 s HOT3D clips
(offline camera: 0.0-0.1 %) and by 4.8 % on the 44 s egocentric desk test video of the project page (up to 1.14x between
the largest and the smallest; offline camera: 3.7 %). The spread reflects how uncertain monocular metric scale is on such
clips rather than an error of its own; the paper reports world metrics as the mean ± SD over 3 runs, which includes it.
A deterministic mode (about 1.3x the camera stages' time) is planned for the next release.

Known issue: on a static camera (a laptop or phone propped up in front of the user) world mode adds nothing and can
drift: the estimated camera wanders (about 20 cm over a minute on such a clip), the hand trails pile up, and the offline
camera's SLAM scale degenerates (`k_ours_over_slam` far from 1, small or large). Use the camera-frame outputs there.
`fah-video --world` prints a warning (and stores `static_camera`, `camera_rotation_deg`, with the offline camera also
`slam_keyframes`) when the camera turns less than 5 degrees over the clip (the offline camera's SLAM trajectory) or 6
degrees (the real-time camera model). On dev clips with camera motion (HOT3D train, EgoDex, EgoStandard dev) the camera
turns at least 6.7 degrees; on tripod clips at most 3.0 (SLAM) / 5.3 (camera model). A camera that slides without
turning (the real-time camera moves 25 cm or more) is not flagged. The real-time margin is small (about 0.7 degrees on
both sides), so the warning is a hint, not a guarantee: a static camera in front of a person who moves a lot can escape
it (both cameras follow the person), and a camera that turns very little can trigger it.

## Evaluation and timing

`python -m bench.world <hand runs> --cams gt,<cams dir>` (WHAM's protocol: W-MPJPE100 / WA-MPJPE100 / RTE on 100-frame
chunks; `--chunk 0`: whole tracks; HOT3D: `bench.run --dataset hot3d` hand runs; see [evaluation.md](evaluation.md)).
The scale correction is fitted with `python -m bench.world_calib fit` on HOT3D *train* clips (never on the benchmark
sequences).
### Statistics

DROID-SLAM and DPVO are not bit-deterministic, so every world cell of the paper is the mean ± sample SD (ddof 1) of a
metric over 3 separate camera runs (3 `bench/world_slam.py` runs for the SLAM rows, `--seed 0` / `1` / `2` for the
camera model; the i-th offline run fuses the i-th SLAM and camera-model runs); each run is scored on its own, predictions
and cameras are never averaged:
```bash
python -m bench.world_reps --chunks 100,0 --out reps.json \
    --row "fah=$RESULTS/hot3d/fah-video|<camera seed 0>|<camera seed 1>|<camera seed 2>" \
    --row "hawor=$RESULTS/hot3d/hawor|<slam run 1>|<slam run 2>|<slam run 3>"
python -m bench.world_bootstrap --dataset hot3d --runs fah0=<hands>,fah1=<hands>,fah2=<hands>,hawor0=<hands>,... \
    --cams fah0=<camera seed 0>,...,hawor0=<slam run 1>,... --avg fah=fah0+fah1+fah2,hawor=hawor0+hawor1+hawor2 \
    --ref hawor --chunk 100 [--common <hands>,<hands>] --out boot.json
```
`bench.world_bootstrap` gives paired sequence-bootstrap intervals (HOT3D: resampled by recording sequence; EgoStandard:
by episode; 2000 resamples, seed 0, 95 %) of the differences between runs, `--avg` recomputing the 3-run mean on every
resample, `--common` restricting all runs to the hand-frames every listed hand run matches, and per-run track statistics
(scored runs, run lengths, share of the error in the 10 worst runs). `python -m bench.world_paper --out paper.json`
recomputes the paper's world cells (the table, its SDs, track statistics and intervals) from the directory layout in
its docstring and exits non-zero if any differs at 2 decimals.

### Camera diagnostics

`python -m bench.world_camdiag --hands <hand run with GT poses> --cams <camera dir> --out <dir> --score` replaces one part
of a predicted camera with ground truth and scores each hybrid with `bench.world` (100-frame chunks and whole tracks): GT
orientations; GT camera centres; the clip's GT metric scale; the GT scale of every 81-frame segment; GT orientations and
segment scale together. They use ground truth, so they are diagnostics only. ### Timing

`scripts/benchmark_world_e2e.py` times the whole pipeline on one GPU (hands and camera as concurrent processes). The
quoted world-space speed (FAH-L with the real-time camera model: 29 fps; the offline camera mode: 7.1 fps; all stages,
desktop RTX 4070 Ti SUPER, 4 EgoStandard test clips, slower of 2 rounds; real time: streaming frame source, slower of
the swap-free runs on this desktop (16 GB RAM; WSL2 VM 14 GB), whose memory is otherwise the limit; two processes are
time-sliced on one GPU (no MPS under WSL2)) is this configuration, without `--mps` on that desktop (`--hand_args`: the
API's world-mode arguments without the mesh fit):
```
HA=$(python -c 'import json; from fah.api import method_args; print(json.dumps(method_args("fah", "world", mesh="none", trt_dir="<engines>")))')
THIRDPARTY=<dir with the DPVO checkout> python scripts/benchmark_world_e2e.py --frames_dir <frames (*.jpg + K.npy)> --frames 900 \
  --mode both --hand_args "$HA" --size 384 --stride 2 --config default --cam_mask --bg_depth --m3d_amp --scale_at 10 \
  --scale_calls 15 --scale_every 30 --reinit_rot 14.2 --reinit_back 8 --reinit_scale 2.0 --trans_gate \
  --head <weights>/shared/camera_motion.pt --no_store --shared_upload --fast_res --depth_model moge2 \
  --depth_py <MoGe-2 env python> --camera_py <DPVO env python> --out world_fps.json
```
(`--mps` runs the stages as clients of a CUDA MPS daemon, where available, instead of time-slicing the GPU.)
Each stage streams its frames from the JPEGs (a decode-ahead ring of `--ahead` 32 frames on a background thread, full
before the timed start, as a camera's capture thread), so no stage holds the clip; `--frame_cache` keeps the earlier
harness (the whole clip in one raw frame file that every stage memory-maps: about 7.5 GB for 900 frames at 1920 x 1456,
which makes a 16 GB machine swap and inflates the per-frame times). `--dump` saves the stages' outputs for parity checks.

The SLAM rows and the offline camera run their stages as separate commands, each in its own environment, and are timed
from timestamped stage logs: pipe every stage command through `python scripts/world_stage_fps.py stamp > <stage>.log`
and run it on `bench/splits/egostandard_timing5.json` (a warm-up clip first, whose gap holds the model loading, then
the 4 timed EgoStandard test clips of 900 frames). `python scripts/world_stage_fps.py fps --stages <stage logs> [--whole
<log>[:<factor>] ...]` computes a stage's wall from the gaps between its consecutive per-clip lines (the warm-up clip
dropped) and a pipeline's fps as the 3,600 timed frames over the sum of its stage walls, plus the whole-command walls of
short CPU steps (scaled by 4/5 when the step also processed the warm-up clip); 2 rounds on an idle GPU, the slower one.
- SLAM rows: WiLoR + SLAM = `bench.run --method wilor` + `bench.run --method hawor` (the SLAM's masks) + the SLAM;
  HaWoR = hawor + SLAM; FAH-L hands + HaWoR's SLAM = the FAH-L video run + hawor + SLAM.
- Offline camera = the FAH-L video run + `bench/world_dpvo.py` + `bench.world_camhead` + `bench/world_slam.py` (stages),
  plus `bench.world_camera calibrate` / `gate` and `bench.world_fuse` at 4/5 and `bench.world_offline` (whole commands).

## Dyn-HaMR (supplement)

The supplement's Dyn-HaMR check uses its released code, fetched at a pinned commit (not part of this repository):
`scripts/dynhamr/setup.sh` (the current version, fa9cd74, and its environment), `scripts/dynhamr/setup_98e1622.sh` (the
paper-time release with its HaMeR fork), `scripts/dynhamr/camfix.sh <tree>` (a copy with the camera-offset fix:
its camera loader adds a constant camera-frame offset to every given pose; known upstream: Dyn-HaMR issues
[#12](https://github.com/ZhengdiYu/Dyn-HaMR/issues/12) and [#25](https://github.com/ZhengdiYu/Dyn-HaMR/issues/25), SLAHMR
[#18](https://github.com/vye16/slahmr/issues/18)). `scripts/dynhamr/run.sh`
runs its stages I-II (`bench/world_dynhamr.py`) on the camera of the WiLoR + SLAM row with the world scale fixed
(`model.opt_scale=False`; the cameras are metric) and writes bench.run pickles plus its own output cameras, scored with
`bench.world`; `bench/world_dynhamr_pa.py` compares the PA-MPJPE of its HaMeR initialiser with its optimised output. Its
stage III (the HMP motion prior) requires 128-frame tracks and is not run.
