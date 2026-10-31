"""MHR-Hand fit (fah/mhr_fit.py): rotation conventions, the closed-form init and known-parameter recovery on a toy
rig with the MHR asset layout (CPU, tiny); the real rig when MHR_HAND_ASSET points to a cached build_asset file."""
import os
import unittest

import numpy as np
import torch

from fah.mhr_fit import MHRHandFitter, euler_xyz, fit_mhr_hand, place_2d, quat_to_rotmat, six_d
from fah.models.heads.fk_eager import compose, euler_xyz_to_quat, quat_rotate
from fah.models.modules.mhr_utils import batch6DFromXYZ


def quat_inv_state(s):
    """Inverse similarity of skeleton states [t, q, s]."""
    t, q, k = s[..., :3], s[..., 3:7], s[..., 7:]
    qi = torch.cat([-q[..., :3], q[..., 3:]], -1)
    return torch.cat([-quat_rotate(qi, t) / k, qi, 1 / k], -1)


def toy_asset(seed=0):
    """Wrist + two fingers (3 + 1 and 2 + 1 euler axes), 2 scale components (wrist scale, bone lengths), 2 shape vectors,
    pose correctives, 72 skinned vertices, 40 'MANO-topology' points and 6 joints regressed from them."""
    g = torch.Generator().manual_seed(seed)
    par = [-1, 0, 1, 2, 3, 1, 5, 6]  # root, wrist, A mcp / pip / tip, B mcp / pip / tip
    nj, P = 8, 10
    dof = [(2, 0), (2, 1), (2, 2), (3, 2), (5, 1), (5, 2), (6, 2)]
    PT = torch.zeros(7 * nj, P)
    for i, (j, ax) in enumerate(dof):
        PT[j * 7 + 3 + ax, i] = 1
    PT[1 * 7 + 6, 7] = 1  # wrist scale
    PT[3 * 7 + 0, 8] = 1  # A pip bone length
    PT[6 * 7 + 0, 9] = 1  # B pip bone length
    A_pose = torch.zeros(7, P)
    A_pose[torch.arange(7), torch.arange(7)] = 1
    A_scale = torch.zeros(2, P)
    A_scale[0, 7], A_scale[1, 8], A_scale[1, 9] = 0.1, 0.5, 0.4
    offsets = torch.tensor([[0, 0, 0], [0, 0, 0], [8, 1, 0], [4, 0, 0], [3, 0, 0], [8, -1.5, 0], [4, 0, 0], [3, 0, 0.0]])
    prerot = euler_xyz_to_quat(0.2 * torch.randn(nj, 3, generator=g))
    levels = [(torch.tensor([1]), torch.tensor([0])), (torch.tensor([2, 5]), torch.tensor([1, 1])),
              (torch.tensor([3, 6]), torch.tensor([2, 5])), (torch.tensor([4, 7]), torch.tensor([3, 6]))]
    loc = torch.cat([offsets, prerot, torch.ones(nj, 1)], -1)
    G = loc.clone()
    for src, tgt in levels:
        G = G.index_copy(0, src, compose(G.index_select(0, tgt), G.index_select(0, src)))
    # vertices along the segments (wrist palm, A mcp / pip, B mcp / pip), each skinned to its segment and its parent
    segs = [(1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)]
    base, sv, sj, sw = [], [], [], []
    for a, b in segs:
        pa, pb = G[a, :3], G[b, :3]
        for k in range(12):
            u = (k + 0.5) / 12
            base.append(pa + u * (pb - pa) + 0.8 * torch.randn(3, generator=g))
            w = min(1.0, 0.6 + u)
            vi = len(base) - 1
            sv += [vi, vi]
            sj += [a, par[a]]
            sw += [w, 1 - w]
    base = torch.stack(base)
    nv = len(base)
    rot = [2, 3, 5, 6]
    a = dict(P=P, nj=nj, n_shape=2, shape_comps=[0, 1], scale_comps=[8, 18], free_scales=False, mp0=torch.zeros(P), A_pose=A_pose,
             A_scale=A_scale, jp0=torch.zeros(7 * nj), J_pose=A_pose @ PT.T, J_scale=A_scale @ PT.T, offsets=offsets, prerot=prerot,
             levels=levels, parents=torch.tensor(par), ibp=quat_inv_state(G), rot_joints=torch.tensor(rot),
             dof_joint=torch.tensor([d[0] for d in dof]), dof_axis=torch.tensor([d[1] for d in dof]), wrist=1,
             part_of_joint=torch.tensor([0, 0, 1, 2, 2, 3, 4, 4]), part_parent=torch.tensor([0, 0, 1, 0, 3]), size_coef=0.1,
             lim_idx=torch.arange(7), lim_lo=torch.full((7,), -1.2), lim_hi=torch.full((7,), 1.2), lim_w=torch.ones(7))
    vs = dict(base=base, shape=0.3 * torch.randn(2, nv, 3, generator=g), sv=torch.tensor(sv), sj=torch.tensor(sj), sw=torch.tensor(sw),
              w0=0.3 * torch.randn(4, 6 * len(rot), generator=g), w2=0.05 * torch.randn(3 * nv, 4, generator=g))
    for p in ("f", "o"):  # the fit set = the output set here
        a.update({p + k: v for k, v in vs.items()})
    W = torch.zeros(40, nv)
    for i in range(40):  # convex combinations of neighbouring vertices
        j = torch.randint(0, nv - 1, (1,), generator=g).item()
        w = torch.rand(1, generator=g).item()
        W[i, j], W[i, j + 1] = w, 1 - w
    Wj = torch.zeros(6, nv)
    for i, j in enumerate([0, 23, 35, 47, 59, 71]):
        Wj[i, j] = 1
    a.update(Wv=W, Wj=Wj, Wv_o=W, Wj_o=Wj)
    return a


def targets(fit, x, R, t):
    with torch.no_grad():
        v, _ = fit.rig.verts(x, fit.B, "f")
        P = torch.einsum("mv,nvc->nmc", fit.B["Wv"], v) @ R.mT + t[:, None]
        J = torch.einsum("mv,nvc->nmc", fit.B["Wj"], v) @ R.mT + t[:, None]
    return P.numpy().astype(np.float64), J.numpy().astype(np.float64)


class ConventionTests(unittest.TestCase):
    def test_euler_inverts_fk_quaternions(self):
        e = (torch.rand(64, 3, generator=torch.Generator().manual_seed(0)) - 0.5) * 2.5
        torch.testing.assert_close(euler_xyz(quat_to_rotmat(euler_xyz_to_quat(e))), e, atol=1e-5, rtol=0)

    def test_six_d_matches_mhr_pose_features(self):
        e = torch.randn(10, 3, generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(six_d(e), batch6DFromXYZ(e))

    def test_place_2d_recovers_the_rigid_placement(self):
        rng = np.random.RandomState(0)
        J = rng.randn(21, 3) * 0.04 + [0.0, 0.0, 0.5]
        K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
        Rt = quat_to_rotmat(euler_xyz_to_quat(torch.tensor([0.1, -0.2, 0.15]))).double().numpy()
        Jt = (J - J[0]) @ Rt.T + J[0] + [0.01, -0.02, 0.03]
        uv = (Jt[:, :2] / Jt[:, 2:]) * 600.0 + [320, 240]
        R, t, c = place_2d(J, uv, K)
        np.testing.assert_allclose((J - c) @ R.T + t, Jt, atol=1e-6)


class ToyRigTests(unittest.TestCase):
    def setUp(self):
        self.fit = MHRHandFitter(toy_asset(), "cpu", iters=15, w_scale=0.0, w_shape=0.0, graph=False, compile=False)

    def test_closed_form_init_is_exact_at_the_reference(self):
        f = self.fit
        R = quat_to_rotmat(euler_xyz_to_quat(torch.tensor([[0.3, -0.5, 1.0]])))
        P, _ = targets(f, torch.zeros(1, f.n), R, torch.tensor([[0.0, 0.0, 0.4]]))
        Pc = torch.as_tensor(P, dtype=torch.float32)
        x0, R0 = f._init(Pc - Pc.mean(1, keepdim=True))
        torch.testing.assert_close(R0, R, atol=1e-4, rtol=0)
        self.assertLess(x0[0, :f.npose].abs().max().item(), 1e-3)

    def test_recovers_known_parameters(self):
        f = self.fit
        g = torch.Generator().manual_seed(2)
        N = 4
        x = torch.zeros(N, f.n)
        x[:, :f.npose] = 0.5 * (torch.rand(N, f.npose, generator=g) - 0.5)
        x[:, f.npose:f.npose + 4] = 0.5 * torch.randn(N, 4, generator=g)
        R = quat_to_rotmat(euler_xyz_to_quat(torch.randn(N, 3, generator=g)))
        t = torch.tensor([0.02, -0.01, 0.4]) + 0.05 * torch.randn(N, 3, generator=g)
        P, J = targets(f, x, R, t)
        out = f(P, J)
        self.assertLess(np.abs(out["verts_mano"] - P).max(), 2e-5)  # metres
        self.assertLess(np.abs(out["joints"] - J).max(), 2e-5)
        np.testing.assert_allclose(out["x"][:, :f.npose], x[:, :f.npose].numpy(), atol=2e-3)
        np.testing.assert_allclose(out["R"], R.numpy(), atol=1e-3)

    def test_closed_form_jacobian_matches_autograd(self):
        f = self.fit
        g = torch.Generator().manual_seed(4)
        x = torch.zeros(3, f.n)
        x[:, :f.npose + 4] = 0.5 * torch.randn(3, f.npose + 4, generator=g)
        v, J = f.rig.verts_jac(x, f.B)
        Ja = torch.func.vmap(torch.func.jacfwd(lambda xx: f.rig.verts(xx[None], f.B, "f")[0][0]))(x)[..., :f.npose + 4]
        torch.testing.assert_close(v, f.rig.verts(x, f.B, "f")[0])
        torch.testing.assert_close(J, Ja.permute(0, 1, 3, 2), atol=1e-6, rtol=1e-4)

    def test_fit_is_continuous_and_deterministic(self):
        f = self.fit
        g = torch.Generator().manual_seed(7)
        x = torch.zeros(3, f.n)
        x[:, :f.npose] = 0.5 * (torch.rand(3, f.npose, generator=g) - 0.5)
        P, J = targets(f, x, quat_to_rotmat(euler_xyz_to_quat(torch.randn(3, 3, generator=g))), torch.tensor([[0.0, 0.0, 0.5]] * 3))
        a, b = f(P, J), f(P, J)
        np.testing.assert_array_equal(a["verts"], b["verts"])
        rng = np.random.RandomState(0)
        c = f(P * (1 + 1e-6 * rng.randn(*P.shape)), J * (1 + 1e-6 * rng.randn(*J.shape)))
        self.assertLess(np.abs(c["verts"] - a["verts"]).max(), 2e-5)  # 1e-6 relative input change -> < 0.02 mm

    def test_segment_sums_match_index_add(self):
        from fah.mhr_fit import MHRHand, segments

        sv = torch.randint(0, 50, (400,), generator=torch.Generator().manual_seed(8))
        y = torch.randn(2, 400, 3)
        torch.testing.assert_close(MHRHand.seg_sum(y, segments(sv, 50)), torch.zeros(2, 50, 3).index_add(1, sv, y))

    def test_closed_form_jacobian_fit(self):
        f = MHRHandFitter(toy_asset(), "cpu", iters=15, w_scale=0.0, w_shape=0.0, jac="analytic", graph=False, compile=False)
        g = torch.Generator().manual_seed(5)
        x = torch.zeros(2, f.n)
        x[:, :f.npose] = 0.5 * (torch.rand(2, f.npose, generator=g) - 0.5)
        x[:, f.npose:f.npose + 4] = 0.5 * torch.randn(2, 4, generator=g)
        P, J = targets(f, x, quat_to_rotmat(euler_xyz_to_quat(torch.randn(2, 3, generator=g))), torch.tensor([[0.0, 0.0, 0.5]] * 2))
        self.assertLess(np.abs(f(P, J)["verts_mano"] - P).max(), 2e-5)


@unittest.skipUnless(os.path.exists(os.environ.get("MHR_HAND_ASSET", "")), "MHR_HAND_ASSET (fah.mhr_fit.build_asset file) not set")
class RealRigTests(unittest.TestCase):
    def test_recovers_a_synthetic_mhr_hand(self):
        a = torch.load(os.environ["MHR_HAND_ASSET"], map_location="cpu", weights_only=False)
        f = MHRHandFitter(a, "cpu", iters=10, w_scale=0.0, w_shape=0.0, graph=False, compile=False)
        g = torch.Generator().manual_seed(3)
        x = torch.zeros(2, f.n)
        lo, hi = f.B["pose_lo"], f.B["pose_hi"]
        x[:, :27] = 0.5 * (lo + 0.05 + (hi - lo - 0.1) * torch.rand(2, 27, generator=g))
        x[:, 27:27 + f.rig.ns] = 0.5 * torch.randn(2, f.rig.ns, generator=g)
        R = quat_to_rotmat(euler_xyz_to_quat(torch.randn(2, 3, generator=g)))
        P, J = targets(f, x, R, torch.tensor([[0.05, 0.0, 0.5], [-0.1, 0.05, 0.6]]))
        out = f(P, J)
        self.assertLess(np.linalg.norm(out["verts_mano"] - P, axis=-1).mean(), 1e-4)
        p = f.params(out, 0)
        self.assertEqual(p["model_params"].shape, (204,))
        self.assertEqual(p["hand_pca"].shape, (54,))

    def test_fit_mhr_hand_left_hands_and_2d_placement(self):
        a = torch.load(os.environ["MHR_HAND_ASSET"], map_location="cpu", weights_only=False)
        f = MHRHandFitter(a, "cpu", iters=6, graph=False, compile=False)
        g = torch.Generator().manual_seed(6)
        x = torch.zeros(1, f.n)
        lo, hi = f.B["pose_lo"], f.B["pose_hi"]
        x[:, :27] = 0.5 * (lo + 0.05 + (hi - lo - 0.1) * torch.rand(1, 27, generator=g))
        P, J = targets(f, x, quat_to_rotmat(euler_xyz_to_quat(torch.tensor([[0.3, 2.5, -0.4]]))), torch.tensor([[0.05, 0.02, 0.5]]))
        m = np.array([-1.0, 1.0, 1.0])  # the same hand as a left hand
        K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
        uv = (J[0, :, :2] * m[:2] / J[0, :, 2:]) * 600.0 + [320, 240]
        r = fit_mhr_hand(f, P[0] * m, J[0] * m, is_right=False, K=K, joints2d=uv)
        self.assertTrue(r["params"]["mirrored"])
        self.assertLess(np.linalg.norm(r["verts_mano"] - P[0] * m, axis=-1).mean(), 2e-4)
        np.testing.assert_array_equal(r["faces"], np.asarray(a["faces"])[:, ::-1])
        self.assertEqual(r["joints_rig"].shape, (21, 3))


if __name__ == "__main__":
    unittest.main()
