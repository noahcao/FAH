# Local configuration

Copy `local.example.json` to `local.json` and edit it. Local paths are relative to the repository;
absolute paths, `~` and environment variables are supported. The local file is ignored by Git.

| Setting | Purpose |
|---|---|
| `data_root` | Prepared datasets, crop pools, contact labels and camera caches |
| `ckpt_root` | Training checkpoint output directory |
| `weights` | Downloaded FAH weights |
| `mano_dir`, `sam_checkpoint` | MANO assets and full SAM 3D Body training checkpoint |
| `model` | `fah-s`, `fah-b`, `fah` (default FAH-L), or `fah-h` |
| `train_processes`, `train_batch_size`, `train_devices` | Your training resource choices; no fixed allocations are supplied |
| `output_dir` | Inference outputs |
| `inference` | `backend` (`torch`, `trt`, `coreml`), `device`, output switches (`dense`, `parts`, `contact`, `masks`), optional `focal` / `fov` |
| `world` | Camera dependencies for world inference; see below |
| `pose_tracks`, `contact_tracks` | Training track specifications accepted by the respective trainers |
| `mesh_data`, `mhr_data` | Extracted fitter training `.npz` files |

Optional keys: `python` selects the FAH interpreter; `sam_python` selects the SAM 3 interpreter;
`sam3_checkpoint` selects its weights; `sam3d` selects inference config/assets;
`model_checkpoint` selects the pose checkpoint used for contact training.
`pose_pools` and `contact_pools` override the public-data defaults with `[pool_name, sampling_weight]` pairs.

For world inference, put `dpvo_dir`, `dpvo_python` and `depth_python` in `world`.
For offline mode, add `camera_version: "offline"`, `droid_dir`, `droid_weights`, `metric3d_dir`,
`metric3d_weights` and `slam_python`. An existing trajectory can be supplied as `cameras`.
Install the separate environments first: [World space](../docs/world.md#requirements).

Training outputs are grouped by model. The shared teacher is in `finetune/teacher`, each model's head initialisation is in
`finetune/<model>/heads`, and its distilled encoder is in `distill/<model>`. Pose training writes
`finetune/<model>/pose/final.pt`; parts write `finetune/<model>/parts/final.pt`.
Contact, video-contact and camera checkpoints have their own output folders.
