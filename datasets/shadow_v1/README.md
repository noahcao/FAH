# shadow_v1: hands with cast shadows

166 images with 322 hands, 274 of them scored. Two types:

- `shadow_A` (78 images): a shadow across the hand (blinds, foliage, window light, other hands or objects).
- `shadow_B` (88 images): the hand next to its own hand-shaped cast shadow, which a detector may take for a second hand.

The set is for testing only.

## Images

Fetch the images with `python datasets/fetch_images.py --set datasets/shadow_v1 --out $DATA_ROOT/shadow_v1 --oi_downloader downloader.py`
(prerequisite: see [../README.md](../README.md)).
`images.jsonl` has one line per image: file, SHA-256, pixel MD5, size, source, licence, author, title, page URL and
the download recipe.

| source | images | licence (per image in `images.jsonl`) | where the file comes from |
|---|---|---|---|
| HaGRID v2 | 51 | HaGRID public licence (a reworking of CC BY-SA 4.0) | the official per-gesture zips (one member each, HTTP range requests) |
| Open Images V7 | 94 | CC BY 2.0 (Flickr photos) | the Open Images copy (official downloader, by `open_images_id`) or the 1024 px Flickr copy, whichever the set used |
| Openverse (Flickr) | 17 | CC BY 2.0 (9), CC BY-SA 2.0 (7), CC0 (1) | the 1024 px Flickr copy, re-encoded as JPEG q95 |
| Wikimedia Commons | 4 | CC BY-SA 4.0 (3), CC BY 2.0 (1) | the 1280 px Commons thumbnail, re-encoded as JPEG q95 |

The HaGRID images show people who are not in our HaGRID training pools. Images were picked by CLIP ranking and then
reviewed by hand.

The paper's shadow_v1 has 170 images (278 scored hands). Four images are withheld because their web sources are unknown. FAH-L
scores the same on both copies at two decimals (recall 1.00, PCK@0.1 0.99, incl. misses 0.99; raw values in
`../expected_fah_l.json`).

## Labels

`annotations.json` uses the bench2d format: one record per image (`id`, `file`, `subset`, `width`, `height`,
`fully_annotated` = true, `hands`). Each hand has:

- `kp2d`: 21 x 2 pixel coordinates in OpenPose order (wrist, thumb1-4, index1-4, middle1-4, ring1-4, pinky1-4);
- `vis` (1 = labelled);
- `is_right`;
- `bbox` (keypoint box, x1 y1 x2 y2);
- `status`;
- `agree`.

How the labels were made:

1. Five hand-pose models predicted every hand, on the original image and on a brightened copy. The predictions were
   clustered per hand.
2. A reviewer picked the best cluster or flagged the hand. The label is the per-joint median of the agreeing candidates
   (within 0.06 of the hand size). `agree` is the number of agreeing models.
3. Hands without a good candidate were labelled by box-prompting our model with both handedness hypotheses.
4. Doubtful hands went to a human pass.

`status` values:

- `picked`: scored.
- `needs_review`: scored, but flagged for a second look.
- `needs_label`: no reliable label. Not scored, and predictions inside its box are not counted as false positives.

The labels come from model candidates. They may favour models in the candidate pool, in PCK and in recall.

## Licence

Our labels: see [../LICENSE.md](../LICENSE.md). Images: their own licences. We do not redistribute them.

- Open Images: "The images are listed as having a CC BY 2.0 license. Note: while we tried to identify images that are
  licensed under a Creative Commons Attribution license, we make no representations or warranties regarding the license
  status of each image and you should verify the license for each image yourself."
  (https://storage.googleapis.com/openimages/web/factsfigures_v7.html)
- HaGRID (https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf), Section 2(b)(1): "Personal non-property
  rights, such as the right to inviolability of the work, and rights of publicity and portrayal, privacy, or other
  similar personal rights, are not granted under this Public License." Fetch HaGRID images from HaGRID's host under
  HaGRID's licence.
- CC BY / CC BY-SA (https://creativecommons.org/licenses/by/2.0/, https://creativecommons.org/licenses/by-sa/4.0/):
  credit the author with a link to the licence and to the photo page (`author`, `license_url`, `page` in
  `images.jsonl`), and say if you changed the image. For CC BY-SA, share adapted images under the same licence.
