# hard_mined_v2: hard in-the-wild hands

169 hands in 167 images:

- 160 images from HaGRID v2 (`hg`), of people outside our training pools;
- 7 photos from Open Images V7 (`oi`).

Our data engine mined the hands: an earlier model of ours was off by more than 0.1 of the hand size on each of them.
Most were cut off by the detector box or got the wrong side. This makes the set a stress test of failure modes, not a
neutral benchmark. It is for testing only.

## Images

Fetch the images with `python datasets/fetch_images.py --set datasets/hard_mined_v2 --out $DATA_ROOT/hard_mined_v2 --oi_downloader downloader.py`
(prerequisite: see [../README.md](../README.md)).

- HaGRID v2 images are read from the official per-gesture zips, one member each, with HTTP range requests.
- Open Images photos are the Open Images copies, fetched by id (`open_images_id`) with the official Open Images downloader.

`images.jsonl` has the licence, author, page URL and SHA-256 of every image. On 2026-10-30 all 167 images came back
byte-identical.

## Labels

`annotations.json` uses the bench2d format, as in [shadow_v1](../shadow_v1/README.md). Only the mined hands are
labelled, so `fully_annotated` is false and precision is not scored.

How the labels were made: candidates from several hand-pose models were clustered per hand, and a reviewer picked one
on review sheets. Wrong fingers were composed from other candidates, and joints that no candidate got right were placed
by hand. `status` records which of these happened (`picked`, `composed`, `fixed`), and `manual` flags hand-placed
joints.

FAH-L scores PCK@0.1 0.98 (recall 1.00), the same as in the paper. The raw value is in `../expected_fah_l.json`.

## Licence

Our labels: see [../LICENSE.md](../LICENSE.md). The images keep their own licences, and we do not redistribute them:

- HaGRID v2: the HaGRID public licence, a reworking of CC BY-SA 4.0. It does not grant personal rights (publicity,
  portrayal, privacy) of the people shown.
- Open Images: CC BY 2.0 per photo.

The licence quotes are in [shadow_v1/README.md](../shadow_v1/README.md).
