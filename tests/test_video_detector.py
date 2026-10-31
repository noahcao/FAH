"""Video detector memory (fah.video.TrackProposals): proposals for confirmed tracks without a detector box, the
constant-velocity prediction, coverage by detector boxes, hands leaving the image; FAHMethod's options default off."""
import inspect
import unittest

import numpy as np

from bench.methods.fah import FAHMethod
from fah.video import TrackProposals, point_box, wrist_dist


def hand(x, y, score=0.9, right=True, prop=False):
    """A hand whose 21 joints span a 20 x 20 px square at (x, y)."""
    g = np.linspace(0, 20, 21, dtype=np.float32)
    kp = np.stack([x + g, y + g[::-1]], 1)
    return dict(keypoints_2d=kp, score=score, is_right=right, **({"vd_prop": True} if prop else {}))


class TrackProposalTests(unittest.TestCase):
    HW = (200, 300)

    def tracked(self, steps=3, dx=5.0):
        tp = TrackProposals(max_missing=3, min_hits=2, enlarge=1.0)
        for k in range(steps):
            tp.update([hand(100 + dx * k, 50)])
        return tp

    def test_confirmed_track_without_a_box_is_proposed_at_its_predicted_position(self):
        tp = self.tracked()
        boxes, sides, ids = tp.proposals(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), self.HW)
        self.assertEqual(len(boxes), 1)
        self.assertTrue(sides[0])
        # last box (x 110) moved by the velocity (EMA of the 5 px steps: 3.75 px / frame)
        np.testing.assert_allclose(boxes[0], [113.75, 50, 133.75, 70], atol=1e-4)
        np.testing.assert_allclose(tp.expected(ids[0]).min(0), [113.75, 50], atol=1e-4)

    def test_no_proposal_when_a_detector_box_covers_the_track_or_the_track_is_new(self):
        tp = self.tracked()
        covered = tp.proposals(np.array([[114, 49, 136, 71]], np.float32), np.array([0.9], np.float32), self.HW)[0]
        self.assertEqual(len(covered), 0)
        weak = tp.proposals(np.array([[114, 49, 136, 71]], np.float32), np.array([0.1], np.float32), self.HW)[0]
        self.assertEqual(len(weak), 1)  # a low-score box does not cover it
        new = TrackProposals(min_hits=2)
        new.update([hand(100, 50)])  # one detector hand: not confirmed yet
        self.assertEqual(len(new.proposals(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), self.HW)[0]), 0)

    def test_tracks_leaving_the_image_are_not_proposed(self):
        tp = TrackProposals(min_hits=2, enlarge=1.0, min_inside=0.6)
        for x in (285, 290, 295):
            tp.update([hand(x, 50)])
        boxes = tp.proposals(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), self.HW)[0]
        self.assertEqual(len(boxes), 0)
        self.assertEqual(len(tp.leaving), 1)

    def test_proposals_move_a_track_but_do_not_keep_it_alive(self):
        tp = self.tracked()
        for k in range(3):  # found again by its proposals only
            h = hand(115 + 5 * k, 50, prop=True)
            tp.update([h])
            self.assertIn("vd_track", h)
        self.assertEqual(len(tp.tracks), 1)
        tp.update([hand(130, 50, prop=True)])  # max_missing frames after the last detector hand: the track ends
        self.assertEqual(tp.tracks, [])
        tp.update([hand(10, 10, prop=True)])  # a proposal never starts a track
        self.assertEqual(tp.tracks, [])
        np.testing.assert_allclose(point_box(hand(10, 10)), [10, 10, 30, 30])


class IdentityTests(unittest.TestCase):
    def test_wrist_distance_is_in_palm_lengths(self):
        palm = np.hypot(9, 9)  # wrist (x, y + 20) to middle MCP (x + 9, y + 11)
        self.assertAlmostEqual(wrist_dist(hand(100, 50)["keypoints_2d"], hand(108, 50)["keypoints_2d"]), 8 / palm, places=6)
        self.assertEqual(wrist_dist(hand(0, 0)["keypoints_2d"], hand(0, 0)["keypoints_2d"]), 0.0)

    def test_match_wrist_joins_tracks_by_identity(self):
        tp = TrackProposals(match_wrist=1.0)
        tp.update([hand(100, 50)])
        tp.update([hand(108, 50)])  # 0.63 palms from the predicted wrist: the same hand
        self.assertEqual(len(tp.tracks), 1)
        tp.update([hand(150, 50)])  # 3.8 palms away: a new hand
        self.assertEqual(len(tp.tracks), 2)


class DefaultsTests(unittest.TestCase):
    def test_video_detector_and_identity_nms_are_off_by_default(self):
        p = inspect.signature(FAHMethod.__init__).parameters
        self.assertEqual((p["vd_propose"].default, p["vd_gate_fill"].default, p["pair_wrist"].default), (False, False, 0.0))
        self.assertEqual((p["vd_dedup_wrist"].default, p["vd_match_wrist"].default), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
