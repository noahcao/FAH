# Acceleration and live camera

[Installation](../README.md#install) · [Usage and outputs](usage.md)

## TensorRT

The reported speeds use TensorRT FP16 engines of the pose encoder and of the detector. Engines are specific to the GPU
model and the TensorRT version, so build them on the target GPU (the conversion needs `pip install -e ".[engine,trt]"`;
nvidia-modelopt may want its own environment, `--trt_python`):

```bash
fah-build-trt --backbone l --calib_images <dir of images with hands>     # -> <weights>/trt/
fah --vid clip.mp4 --trt --output_dir output/fah
```

`fah-build-trt` runs `scripts/build_pose_engine.py` (export + build) and `scripts/build_det_engine.py`. When
`<weights>/<model>/contact.pt` has a contact branch (its own copies of the encoder's last blocks), the pose engine also
outputs the branch's input; the same engine serves runs with and without contact (`--no_contact_branch`: the engine
without it, for use without contact). Build the engines after updating `contact.pt`.

Without engines the same models run in PyTorch (CUDA graphs on GPU; Apple MPS and CPU are supported, slower).
PyTorch MPS defaults to float32: bfloat16 can abort inside Metal matrix multiplication on macOS 15 / PyTorch 2.15,
and float16 overflows in ViT-L. Core ML pose exports default to float32; their detector remains fp16.

## Apple silicon (Core ML) and live camera

Real-time hand estimation on your Mac's camera (Apple silicon, macOS 15+), in one script:

```bash
git clone https://github.com/noahcao/FAH.git
cd FAH
scripts/mac_live.sh        # first run: --weights <dir> (unless in ~/.cache/fah/weights) --mano ~/Downloads/mano_v1_2.zip
```

The script prepares `.venv`, SAM files, converted MANO models and Core ML exports, then starts
FAH-S in video mode with mesh fitting, contact and parts off. Supply the FAH weights, accept the SAM
licence and register for MANO first ([installation](../README.md#install)). macOS requests camera access.
Use `FAH_PIP=1` to install with pip instead of uv; other arguments pass to `fah-live`.

Core ML runs pose on the GPU, detection on the Neural Engine and mesh fitting on CPU.
The builder records a `.fah-build.json` sidecar for each export. Reuse requires matching precision,
checkpoint and upstream-asset content hashes, batch sizes or detector shapes, exporter code and toolchain versions,
and the package content hash. Changes trigger rebuilding of the affected exports; packages without metadata rebuild
once. Metadata is written only after a successful build with unchanged inputs. `--force` still rebuilds everything.
Install `pip install -e ".[coreml]"` and build exports once per model. The part-free export is the default;
`--parts` or live key `m` loads the full model. Unsupported aspect ratios are padded; build an exact
shape with `--frames <W>x<H>`. Packages compile to adjacent `.mlmodelc` files for faster loading.
The GPU cache in `~/Library/Caches/python/com.apple.e5rt.e5bundlecache` can be cleared to reclaim disk.

```bash
fah-build-coreml --backbone s --pose_precision float32 --parity_frames <dir of <stream>_<index>.jpg frames>   # -> <weights>/coreml/ + a parity report
fah --vid clip.mp4 --backbone s --coreml --device mps --output_dir output/fah        # PyTorch MPS (float32) if the exports are missing
fah-live                                              # live camera, FAH-S real-time preset (online video mode, no mesh fit, fingertip step on)
fah-live --backbone l --mesh_fit 1 --record out.mp4   # keys: q quit, k keypoints, m masks / parts, f mesh fit, v video mode, s snapshot
fah-live --video_mode 0                               # per-frame image mode (default: the online video refiner, as --mode video)
```

`fah-live` also runs on Linux / CUDA (PyTorch) and on a video file (`--video clip.mp4`); `--camera`, `--fov` / `--focal`
and `--record` choose the source, the intrinsics and an overlay recording.

Instance masks (`det_masks=True` / `--masks`) on the Core ML path are sampled from the detector's mask prototypes with two
small matrix products per box instead of `roi_align` at frame resolution (the same masks up to float rounding; a few
differ by a pixel). For live use, `det_masks="crop"` returns `mask_det_crop` = (x0, y0, the boolean mask inside the box)
instead of the full-frame RLE `mask_det`, skipping the full-frame mask and its encoding (`bench.seg.rle.paste_crop` /
`encode_crop` rebuild the full mask / its RLE).

Core ML pose exports default to float32 because fp16 conversion can produce nonfinite outputs on
some macOS/PyTorch combinations. Rebuild with `fah-build-coreml --force --pose_precision float32`.
Use `--pose_precision float16` only after validating parity on your toolchain. Measure throughput
on your own exported model and target device.

Known issue: FAH-L occasionally aborts on its first Core ML GPU prediction with
`MPSGraphTensorData.mm:223: failed assertion shape.count = 0 != strides.count = 3` (exit 134).
The CLI retries once during startup; custom callers should restart. This can also occur when loading
part maps with key `m`. FAH-S was unaffected in our runs.
