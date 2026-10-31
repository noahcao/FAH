"""The MANO head's forward kinematics (fah.models.heads.mano_head._rigid_transform, used in place of smplx's
batch_rigid_transform) returns exactly smplx's posed joints and skinning transforms (needs smplx)."""
import importlib.util
import unittest

import torch

from fah.fast_mano import rodrigues
from fah.models.heads.mano_head import _rigid_transform

MANO_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]


@unittest.skipUnless(importlib.util.find_spec("smplx"), "needs smplx")
class ManoFKTests(unittest.TestCase):
    def check(self, device):
        from smplx import lbs

        if lbs.batch_rigid_transform is _rigid_transform:  # patched by a MANO head earlier in this process
            importlib.reload(lbs)
        ref = lbs.batch_rigid_transform
        g = torch.Generator().manual_seed(0)
        parents = torch.tensor(MANO_PARENTS, device=device)
        for B in (1, 2, 7):
            rot = rodrigues(torch.randn(B * 16, 3, generator=g) * 1.5).reshape(B, 16, 3, 3).to(device)
            joints = (torch.randn(B, 16, 3, generator=g) * 0.05).to(device)
            a = ref(rot, joints, parents)
            b = _rigid_transform(rot, joints, parents)
            for x, y in zip(a, b):
                self.assertEqual(x.shape, y.shape)
                self.assertTrue(torch.equal(x, y))

    def test_matches_smplx_on_cpu(self):
        self.check("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_matches_smplx_on_cuda(self):
        self.check("cuda")


if __name__ == "__main__":
    unittest.main()
