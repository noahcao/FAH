# Real to sim

An application of the video hands (supplementary material, project-page videos): a hand-object video is replayed in a
physics simulator with FAH's hands and an object reconstructed from the same video. The success rates are application
results: two replay steps act on contact (below), so they do not measure contact accuracy.

Code: `scripts/real2sim/` (run from the repository root). Clips and protocol of the reported study:
`bench/splits/real2sim.json`. One driver for the whole study: `scripts/real2sim/run_study.sh`.

## Pipeline

| stage | script | output (in the clip directory) |
|---|---|---|
| clip | `prep_clip.py` (DexYCB tar, HOT3D-Clips tar or a video) | `frames/`, `clip.json` (K, object name, source), `gt.npz` (DexYCB), `cams.npz` (HOT3D headset poses) |
| depth | `depth.py` (MoGe-2) | `depth.npz` (metric depth, K) |
| hands | `hands.py` (`fah.FAH`, video mode) | `hands.pkl` (per frame: MANO mesh / joints in the camera frame, detector masks) |
| object masks | `segment.py` (SAM 3) | `masks.npz` |
| object mesh | `reconstruct_s3do.py` (SAM 3D Objects) | `recon_s3do/mesh.glb`, `layout.json` |
| object track | `track.py` | `track.npz`, `recon_s3do/mesh_metric.glb` |
| replay | `sim.py` (MuJoCo) | `sim_<tag>/metrics.json` (the success rule's terms), panel videos with `--render 1` |
| scoring | `repeats.py`, `candidates.py` | per-clip pass counts, rates per source and seen / unseen, failure stages |
| figures | `compose.py` (page videos), `still.py` (unlabelled stills) | |

- **Object masks.** SAM 3 with a text prompt: the dataset's own object name (`--text_rule dataset`: the YCB name for
  DexYCB, the table `segment.HOT3D_TEXT` for HOT3D). The prompt frame is searched every 5 frames for the most confident
  instance that no FAH hand mask touches (then the name's last word; `--search 5`), the masks are propagated both ways, and
  identity switches of the tracker are re-linked to the instance overlapping the last kept mask (`--relink 0.2`).
- **Object mesh.** SAM 3D Objects on one frame: the largest mask x score before a hand touches the object
  (`--frame -1`), with the MoGe-2 pointmap; mesh, layout and a baked texture.
- **Object track.** Silhouette terms from the reconstruction's layout (`--init layout`); the object stays static until a
  hand mask touches it; FAH's metric hand sets the object's scale (one factor per clip, `hand_metric_scale`).
  `--grasp_hold 1`: a second pass in which, while a hand holds the object, the pose is predicted from the hand's palm
  motion and the refit on the masks is kept only when the masks support it (re-acquired), so the object follows the hand
  through occlusion. HOT3D clips use both hands (`--two_hands 1`) and the headset poses for the camera motion.
- **Replay.** The object is a free body of convex parts (CoACD); each hand is 16 convex MANO parts welded to targets that
  follow the video hands. A contact optimisation (`contact.py`, `--contact 1`) translates the hand per frame until its parts
  touch the object; the object is held in place until it moves in the video (`--hold_until_lift 1`); a finger controller
  then closes the grasp: joint-space closing (`--ctrl_flex 1`: each finger's MCP / PIP / DIP joint flexes, about axes from
  the estimated mesh, until a segment distal of it touches the object, by at most 30 degrees) or the Cartesian controller
  (`--ctrl_flex 0`: finger segments within 2 cm move onto the surface by at most 15 mm); parts deeper than 4 mm back out
  (`--ctrl_pen 0.004`). Gravity: the support plane fitted on the depth, or the headset's world frame for moving cameras
  (`--gravity headset`).
- **Success rule** (`sim.page_success`, fixed before any clip of the study ran): the video lifts the object by at least
  5 cm (otherwise the clip counts as a failure), and the simulation (1) lifts it by at least 5 cm, (2) holds it (at least
  3 cm up and touching the hand) in at least 80% of the frames in which the video has it lifted and for at least 0.5 s,
  (3) still holds it in the last 0.3 s, (4) keeps the 95th percentile of the deepest hand-object penetration at or below
  10 mm and (5) keeps its centre within 5 cm of the video track (median over the held frames). A relaxed variant allows
  8 cm in (5) (`candidates.secondary`).
- **Robot hand (optional).** `sim.py --robot leap|allegro|shadow|inspire` retargets the hand to a robot hand with
  dex-retargeting and replays it kinematically (not part of the study).

## Requirements

Each external component is fetched under its own licence; none is vendored here.

- The FAH environment (README) plus `pip install mujoco coacd trimesh "imageio[ffmpeg]" scipy` (checked with mujoco
  3.17.0, coacd 1.0.14, trimesh 5.2.0, numpy 2.5.4, torch 2.11). MANO: `MANO_DIR` (default `$FAH_HOME/mano`). Replays with
  `--render 0` (metrics only) need no GL context; `--render 1` renders offscreen with `MUJOCO_GL=egl` (a GPU).
- [MoGe-2](https://github.com/microsoft/MoGe) (`pip install -e ".[world]"`, docs/world.md); `MOGE_CKPT` = a local
  `model.pt`, else `Ruicheng/moge-2-vitl-normal` from the hub.
- [SAM 3](https://github.com/facebookresearch/sam3) in its own environment (`SAM3_PY`); checkpoint `SAM3_CKPT`
  (default `$FAH_HOME/sam3/sam3.pt`). SAM License (gated on the hub).
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) in its own environment (`SAM3DO_PY`, its README;
  texture baking needs nvdiffrast and the mip-splatting rasteriser it lists), checkout `SAM3D_OBJECTS_REPO`, checkpoints'
  `pipeline.yaml` at `SAM3D_OBJECTS_CONFIG`; run with `LIDRA_SKIP_INIT=true` and `CONDA_PREFIX` = the CUDA toolkit.
  SAM License (gated on the hub). Publications using it acknowledge the use of SAM Materials.
- Optional: [dex-retargeting](https://github.com/dexsuite/dex-retargeting) and the robot descriptions of
  [dex-urdf](https://github.com/dexsuite/dex-urdf) (`DEX_URDF_DIR`, each robot under its own licence) for `--robot`.
- Data: DexYCB (CC BY-NC 4.0) converted with `bench.datasets.build_dexycb`
  (`$DATA/dexycb/clips/<subject>/<sequence>_<camera>.tar`), HOT3D-Clips (`$DATA/hot3d/train_aria/clip-*.tar`, the
  dataset's licence). MuJoCo is Apache-2.0, CoACD MIT, MoGe MIT.

## The study

`bench/splits/real2sim.json`: 8 dev clips (the pipeline was developed on these only, then frozen; excluded from every
rate) and 54 evaluation clips: 30 DexYCB (third person; 15 are DexYCB s0-train sequences, in FAH's training data, listed in
`fah_seen`; the other 15 are s0-test sequences) and 24 HOT3D (egocentric; HaWoR validation sequences, never trained on).
Clips follow the fixed rules in the split file. Six HOT3D clips (`screened`) were manually screened from their input
frames; report results both with and without this subset. Hands:
FAH-L (`--model fah`) in video mode, with the detector masks; the study ran the TensorRT engines of
`fah-build-trt` (`TRT=<engine dir>`), PyTorch gives slightly different numerics.

The replay is not bit-reproducible across runs, so each clip is replayed three times per controller with CoACD seeds
0 / 1 / 2, and a clip passes if the rule holds in at least two. Two things vary: the convex parts (CoACD; the same seed
gives the same parts) and an unseeded surface sample in the test of when the hand has reached the object, which sets the
anchor release (`--sample_seed`, default -1 = unseeded as in the study; `--sample_seed 0` makes a replay repeatable in one
environment). The primary controller (joint-space closing) was fixed before the evaluation; on the dev clips it was at
parity with or slightly below the Cartesian controller, which is reported on the same inputs.

```bash
export R2S=work/real2sim DATA=/data SAM3_PY=<sam3 env>/bin/python SAM3DO_PY=<sam3do env>/bin/python TRT=<engines>
for g in eval_old eval_new; do
  for s in prep hands seg recon track; do GROUP=$g bash scripts/real2sim/run_study.sh $s; done
  for c in joint cartesian; do for r in 1 2 3; do
    GROUP=$g CTRL=$c REP=$r bash scripts/real2sim/run_study.sh sim
    GROUP=$g CTRL=$c REP=$r bash scripts/real2sim/run_study.sh diag      # DexYCB: FAH hand + GT object, GT hand + GT object
  done; done
done
GROUP=eval_old bash scripts/real2sim/run_study.sh score                 # -> $R2S/repeats_{joint,cartesian}.json + tables
python scripts/real2sim/candidates.py --root $R2S --candidates <clips> --run sim_joint --diag _joint --split bench/splits/real2sim.json
```

`GPUS="0 1 2 3"` spreads the clips over GPUs (the stages run one clip at a time; independent clips can run in parallel).
Replay tags: `sim_joint` / `sim_joint_r2` / `sim_joint_r3` (joint space), `sim_cart[_r2|_r3]` (Cartesian); diagnostics
`sim_diag_gtobj_<tag>`, `sim_diag_gtgt_<tag>`.

Reported results (strict rule, pass in >= 2 of 3 replays; "1 run": the seed-0 replay only):

| clips | joint space: strict | 8 cm | 1 run | Cartesian: strict | 8 cm | 1 run |
|---|---|---|---|---|---|---|
| all (54) | 7 | 9 | 7 | 5 | 5 | 4 |
| without the 6 screened (48) | 7 | 9 | 7 | 5 | 5 | 4 |
| DexYCB, in FAH's training data (15) | 3 | 5 | 4 | 2 | 2 | 1 |
| DexYCB, unseen (15) | 4 | 4 | 3 | 3 | 3 | 3 |
| HOT3D, egocentric (24) | 0 | 0 | 0 | 0 | 0 | 0 |
| of these, manually screened (6) | 0 | 0 | 0 | 0 | 0 | 0 |

GT diagnostics on the 25 DexYCB clips with a reconstruction (same replay and repeats), FAH hand + GT object / GT hand +
GT object: joint space 5 / 7 of 15 seen and 5 / 3 of 10 unseen; Cartesian 6 / 7 and 4 / 5. Passing clips (joint space,
passes of 3): mustard 3, marker 3, drill2 3, banana2 3, soupcan2 3, n_chefcan_l 3, n_woodblock 2. n_woodblock passes
with the reconstructed object but not with the GT object.

## Check

The ported code, in a fresh environment (versions above), on stored study inputs (`track.npz`, the reconstruction, the
contact-optimisation result) with `--render 0`:
- marker (joint space, seed 0, an all-pass clip): the rule's terms reproduce exactly (lift 280.0 mm, held 100% / 0.90 s,
  p95 penetration 5.5 mm, centre 41.0 mm, pass), with the study's convex parts (`--reuse_assets 1`) and with freshly
  decomposed ones (seeds 0 and 1);
- n_woodblock (joint space, the clip with 2 / 3): the fresh convex parts are identical to the study's for all three seeds;
  seed 0 gives the stored outcome (fail, p95 penetration 11.2 mm) in 4 of 5 runs and a pass once, when the unseeded
  surface sample found the hand one frame earlier (frame 24 instead of 25); seeds 1 and 2 give the stored outcome (pass) with
  terms within millimetres; with `--sample_seed 0` the runs are identical.
- End to end (DexYCB marker, release code and weights, one GPU): frames, depth, the SAM 3 masks (SAM 3 at commit f1647e8)
  and the SAM 3D Objects layout (at f91db41) reproduce the study's outputs exactly, with the mesh within 0.3 mm (posed
  Chamfer). The hands agree to within 1.0 mm (joints; median 0.09 mm), with the same detections and identical masks. The
  object track differs from the study's by 17 mm (median): the tracker's own spread across environments, since
  re-tracking the study's own inputs gives the same difference and two runs in one environment differ by 4 mm. It is
  closer to the ground truth (centre error 16-18 mm vs 27 mm). All 18 replays (3 tracks x 2 controllers x 3 seeds) pass,
  as stored (3 / 3 for each controller).

## Videos and stills

`sim.py --render 1 --reuse_assets 1 --tag <tag>` re-renders a scored replay with the convex parts that replay used (its
`sim_<tag>/assets`), so the video comes from the run whose metrics are reported (of the 7 re-rendered passes of the
study, 5 reproduced the rule's terms exactly and 2 within millimetres, with the same outcome; add `--sample_seed` for exact
re-renders of new runs); `compose.py --panels input,overlay,phys --cols 3 --width 1440 --out page_1440.mp4` tiles
input | FAH hands + object | MuJoCo (`--relabel "overlay=..."` rewrites a panel's label). `still.py` writes unlabelled stills of one frame (kinematic placement, no contact optimisation, no physics).
