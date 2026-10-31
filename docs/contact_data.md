# Contact data: HOI4D, ARCTIC, MOW

The contact heads train on HOT3D, DexYCB, HO-3D, OakInk2, InterHand2.6M (contact-free negatives), ARCTIC, HOI4D and MOW
(training.md), and are evaluated on HOT3D, DexYCB, HOI4D, ARCTIC and MOW (evaluation.md). This page covers the three sets
with their own builders here. None of them is redistributed: fetch each from its authors under its licence, then build the
crops / labels / clips locally. Everything below writes to one contact data root (`--out`, default `$CONTACT_ROOT` or
`data/contact`). The builders need `pip install .[contact]` (Open3D, libigl, trimesh) and the MANO models (`$MANO_DIR`).

## Label rule

`train/contact_sdf.py`: the signed distance (mm) of each ground-truth MANO vertex to the ground-truth object surface, exact
point-to-triangle (Open3D), negative inside (generalised winding number > 0.5, libigl, tested on every point inside the
object's box); a vertex is in contact iff it is < 10 mm; distances are stored clipped to +-500 mm, so the 5 / 20 / 35 mm
channels come from the same labels.

## HOI4D

[HOI4D](https://hoi4d.github.io) (CC BY-NC 4.0). Download into one directory (`--raw`): `HOI4D_Hand_pose.zip`,
`HOI4D_annotations.zip`, `HOI4D_CAD_Model_for_release` (unzipped), `camera_params.zip` and `HOI4D_release.zip` (RGB videos).
The two larger zips may also be given unzipped as the directory of the same name.

    python -m train.build_hoi4d_contact pool --raw $DATA_ROOT/hoi4d --out $DATA_ROOT/contact --split train --stride 4
    python -m train.build_hoi4d_contact pool --raw $DATA_ROOT/hoi4d --out $DATA_ROOT/contact --split test --stride 5
    python -m train.build_hoi4d_contact video --raw $DATA_ROOT/hoi4d --out $DATA_ROOT/contact --split train,test
    python -m train.build_hoi4d_contact check --raw $DATA_ROOT/hoi4d --n 20 --out /tmp/hoi4d_check   # overlay sheet

- Rigid categories with hand and object pose only (ToyCar, Mug, Bottle, Bowl, Kettle, Knife, Chair). Split
  `bench/splits/hoi4d_contact.json`: test = subject H3, train = H1 / H2 / H4.
- Hands: HOI4D's fits use manopth (flat_hand_mean, 45 axis-angle) with MANO_LEFT.pkl's left hand as shipped (no
  left-shapedirs fix). This convention reproduces HOI4D's 2D keypoints and agrees with its 2D hand masks; the "fixed"
  alternative moves left hands by ~36 mm. Keep it. Left hands only occur in Chair sequences.
- Chair: its ground truth is unreliable (frequent hand-object penetration). The contact recipe drops Chair crops and crops
  whose hand penetrates its object by more than 15 mm from training, and HOI4D test is reported without Chair:
  `bench/splits/hoi4d_contact_test_nochair.json` (149 of 176 test sequences; image test: drop the crops with
  `obj == "Chair"`).

## ARCTIC

[ARCTIC](https://arctic.is.tue.mpg.de) (MPI-IS non-commercial research licence; account needed). Clone the
[ARCTIC repo](https://github.com/zc-alexfan/arctic) and fetch + unpack the data with its scripts (`raw_seqs`, `meta`,
`splits_json`, `cropped_images_zips`); `--arctic` is that checkout (the builders import its MANO / object layers).

    python -m train.build_arctic_contact --arctic $DATA_ROOT/arctic --out $DATA_ROOT/contact --split train --stride 8
    python -m train.build_arctic_contact --arctic $DATA_ROOT/arctic --out $DATA_ROOT/contact --split val --stride 8
    python -m train.build_arctic_video --arctic $DATA_ROOT/arctic --out $DATA_ROOT/contact --split train
    python -m train.build_arctic_video --arctic $DATA_ROOT/arctic --out $DATA_ROOT/contact --split val

- Egocentric view, the released cropped frames undistorted to an exact pinhole view; hands from ARCTIC's own MANO layers
  (left hand as shipped: applying the smplx left-shapedirs fix to ARCTIC's fits moves left hands by up to 22 mm);
  the object articulated per frame. Protocol p1: train -> pool / training clips, val -> the held-out test set
  (`test/arctic_val`).

## MOW

MOW images and poses ([zhec.github.io/rhoi](https://zhec.github.io/rhoi)) with HACO's preprocessed 35 mm contact labels,
masks and splits (HACO's release). `train/build_mow_contact.py` builds the training pool (`pool`: the FAH detection most
inside HACO's hand + object mask, labelled with HACO's 35 mm contact; it supervises only the 35 mm output) and the test
predictions (`preds --model fah | haco`); `poses` turns MOW's own MANO fits into a pose crop pool (f = max(H, W),
principal point at the centre of the bottom / right zero-padded square, the fit's hand scale dropped). The evaluation
uses HACO's protocol: per-image F1 at 35 mm, mean over the 92 test images, HACO cropped by its mask box x 1.3 at its 0.52
threshold. Paired deltas with an image bootstrap:

    python scripts/contact_mow_pairs.py --mow $DATA_ROOT/mow --preds $DATA_ROOT/contact/test/mow \
        --arms fah=<fah tag>@0.58,haco=haco@0.52 --pairs fah-haco

## HACO baseline

`bench/methods/haco.py` runs the official [HACO](https://github.com/dqj5182/HACO_RELEASE) ViT-H (`$THIRDPARTY/HACO`, its
environment and `release_checkpoint/`). `HACOMethod(crop="hand2x").predict(image, boxes, is_right)` on given boxes
(ground truth: HOT3D / DexYCB, threshold 0.56), `HACOMethod()(image)` with WiLoR's hand detector (the WiLoR / HaMeR
adapters' settings) so all baselines of a video share one detection; MOW: `HACOMethod(crop="mask13", thr=0.52,
jpeg_quality=None).predict(image, [box], [True], windows=[mask_box(mask)])` with HACO's hand + object mask.
