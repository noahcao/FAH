# low_light: synthetic low-light copy of bench2d_v2

The same 6,364 images and 8,582 labels as [bench2d_v2](../bench2d_v2/README.md), with every image darkened by
`bench.bench2d.make_dark --level dark`. The transform runs in linear light:

1. exposure 0.05-0.15x;
2. Poisson shot noise and Gaussian read noise at the reduced signal;
3. weak gain (0.8-1.5x) with a slight colour cast;
4. sRGB gamma;
5. a slight blur on half of the images;
6. JPEG q60-85, stored at q95.

The random numbers are seeded by the image id, so the copy is deterministic.

```bash
python -m bench.bench2d.make_dark --src $DATA_ROOT/bench2d_v2 --dst $DATA_ROOT/bench2d_v2_dark --level dark
python datasets/check_set.py --ann $DATA_ROOT/bench2d_v2_dark/annotations.json --ref datasets/low_light/fingerprints.json --images
```

On 2026-10-30, regenerating 200 random images from the paper's bench2d_v2 copy gave byte-identical files (OpenCV 5.0.0,
NumPy 2.5.2). Other OpenCV / libjpeg builds may differ in the last bits of the JPEG files. `check_set.py` then reports
"image differs" while the labels stay the same.

Expected FAH-L values on this synthetic copy: recall 0.97, PCK@0.1 0.81, incl. misses 0.78. The raw values (0.9654 /
0.8067 / 0.7791) are under `low_light.expected` in `../expected_fah_l.json`.

The paper's low-light cells (0.96 / 0.81 / 0.78) were scored on a larger set: this synthetic copy plus 19 real dark
photos (50 hands, one unlabelled). We may not redistribute those photos, so they are not here.

Licence: as bench2d_v2. The darkened images are adaptations of the upstream images.
