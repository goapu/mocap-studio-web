"""Real-time pipeline: geometry, temporal model, decoding and tracking."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.realtime.engine import PacketBuffer, encode_packet  # noqa: E402
from backend.realtime.pose2d import simcc_decode  # noqa: E402
from backend.realtime.skeletons import COCO17, HALPE26  # noqa: E402
from backend.realtime.synthetic import ring_calibration, running_motion  # noqa: E402
from backend.realtime.temporal import BoneLengths, OneEuro  # noqa: E402
from backend.realtime.tracker import CameraTracker, choose_box, keypoint_box  # noqa: E402
from backend.realtime.triangulation import (  # noqa: E402
    CONFLICT,
    OBSERVED,
    CameraRig,
    repair_left_right,
    triangulate,
)


@pytest.fixture
def rig():
    cal = ring_calibration(4)
    for i, cam in enumerate(cal["cameras"].values()):
        cam["dist"] = [0.04 * (i % 2), -0.01, 0.001, -0.001, 0.0]
    return CameraRig.from_calibration(cal)


def points(n=17, seed=0):
    return np.random.default_rng(seed).normal([0, 0, 1.0], [0.3, 0.3, 0.5], (n, 3))


def test_triangulation_is_exact_with_distortion(rig):
    X = points()
    uv, _ = rig.project(X)
    res = triangulate(rig, uv, np.full(uv.shape[:2], 2.0))
    assert np.abs(res["X"] - X).max() < 1e-6
    assert (res["status"] == OBSERVED).all()
    # Covariance is a valid 3×3 SPD matrix with sub-centimetre sigma.
    sig = np.sqrt(np.diagonal(res["cov"], axis1=1, axis2=2))
    assert np.all(sig > 0) and np.all(sig < 0.01)


def test_bad_view_is_dropped_not_the_joint(rig):
    X = points()
    uv, _ = rig.project(X)
    uv[2, :, 1] += 80  # one camera is badly wrong for every joint
    res = triangulate(rig, uv, np.full(uv.shape[:2], 2.0))
    assert np.abs(res["X"] - X).max() < 1e-4
    assert not res["used"][2].any() and res["used"][[0, 1, 3]].all()


def test_two_conflicting_views_are_flagged(rig):
    X = points(3)
    uv, _ = rig.project(X)
    uv[2:] = np.nan
    uv[1, :, 1] += 40  # across the (horizontal) epipolar lines
    res = triangulate(rig, uv, np.full(uv.shape[:2], 2.0))
    assert (res["status"] == CONFLICT).all()


def test_missing_views_give_null(rig):
    X = points(2)
    uv, _ = rig.project(X)
    uv[1:] = np.nan
    res = triangulate(rig, uv, np.full(uv.shape[:2], 2.0))
    assert np.isnan(res["X"]).all()


def test_left_right_swap_is_repaired(rig):
    _, gt = running_motion(0.5, 30)
    uv, _ = rig.project(gt[5])
    bad = uv.copy()
    for a, b in [(11, 12), (13, 14), (15, 16)]:
        bad[1, [a, b]] = bad[1, [b, a]]
    fixed, _, swapped = repair_left_right(
        bad, np.full(uv.shape[:2], 2.0), uv, COCO17.swap_groups
    )
    assert swapped[1, 0] and not swapped[[0, 2, 3]].any()
    assert np.allclose(fixed, uv)


def test_temporal_model_beats_per_frame_solution():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from benchmark_accuracy import run

    res = run(cams=4, fps=60, duration=3, old=False)
    raw = res["Real-time triangulation"]
    live = res["+ live Kalman filter (display)"]
    final = res["+ RTS smoothing (saved result)"]
    assert final["coverage"] == 100
    assert live["mpjpe"] < raw["mpjpe"]
    assert final["mpjpe"] < 0.6 * raw["mpjpe"]
    assert final["jitter"] < 0.3 * raw["jitter"]
    assert final["mpjpe"] < 8.0  # millimetres


def test_bone_constraint_restores_lengths():
    X = np.zeros((1, 17, 3))
    X[0, 5] = [0, 0, 1.5]
    X[0, 7] = [0, 0, 1.0]  # 0.5 m upper arm
    X[0, 9] = [0, 0, 0.7]
    bones = BoneLengths(COCO17)
    lengths = np.full(len(bones.edges), np.nan)
    lengths[0] = 0.3  # shoulder→elbow
    out = bones.apply(X, lengths)[0]
    assert np.isclose(np.linalg.norm(out[7] - out[5]), 0.3)
    # Forearm moved rigidly with the elbow.
    assert np.isclose(np.linalg.norm(out[9] - out[7]), 0.3)


def test_one_euro_removes_jitter_and_follows_steps():
    f = OneEuro()
    rng = np.random.default_rng(0)
    xs = [f(np.array([100.0]) + rng.normal(0, 2, 1), t / 60) for t in range(120)]
    assert np.std(xs[60:]) < 1.0
    ys = [f(np.array([300.0]), (120 + t) / 60) for t in range(30)]
    assert abs(ys[-1][0] - 300) < 5


def test_simcc_subbin_decoding():
    bins = np.arange(384, dtype=float)
    peak = 201.3
    simcc = np.exp(-0.5 * ((bins - peak) / 4) ** 2)[None, None]
    loc, val = simcc_decode(simcc)
    assert abs(loc[0, 0] - peak / 2) < 0.02
    assert val[0, 0] > 0.95


def test_skeleton_definitions_are_consistent():
    for sk in (COCO17, HALPE26):
        assert sorted(sk.flip_index()) == list(range(sk.size))
        assert all(max(b) < sk.size for b in sk.bones)
        desc = sk.descendants()
        for parent, child in sk.limb_tree:
            assert child in desc[child] and parent not in desc[child]


def test_tracker_follows_fast_motion_and_detects_loss():
    tr = CameraTracker("cam1", (1280, 720), 17, det_interval=10)
    assert tr.wants_detection(0)
    tr.seed(np.array([500, 100, 700, 600, 0.9]))
    assert tr.crop_box() is not None
    base = np.array([[600 + 10 * np.sin(j), 150 + 25 * j] for j in range(17)], float)
    scores = np.full(17, 0.9)
    for f in range(6):
        kp = base + [40 * f, 0]  # 40 px per frame sideways sprint
        assert tr.update(f, None, kp, scores)
    box = tr.crop_box()
    next_kp = base + [40 * 6, 0]
    assert box[0] <= next_kp[:, 0].min() and box[2] >= next_kp[:, 0].max()
    assert not tr.update(6, None, next_kp, np.full(17, 0.05))
    assert not tr.tracking


def test_choose_box_prefers_reference_overlap():
    boxes = np.array([[0, 0, 400, 700, 0.99], [900, 100, 1100, 500, 0.8]])
    assert choose_box(boxes)[0] == 0
    assert choose_box(boxes, reference=np.array([880, 90, 1090, 520]))[0] == 900
    assert keypoint_box(np.full((17, 2), np.nan)) is None


def test_packet_buffer_streams_and_skips_ahead():
    buf = PacketBuffer(capacity=3)
    for i in range(5):
        buf.push(encode_packet({"frame": i}, [b"\xff\xd8jpeg"]))
    buf.close()
    got = list(buf.iterate(0))
    assert len(got) == 3  # oldest two were dropped for a slow viewer
    import json
    import struct

    body = got[0][4:]
    head_len = struct.unpack(">I", body[:4])[0]
    header = json.loads(body[4 : 4 + head_len])
    assert header["frame"] == 2 and header["jpeg_sizes"] == [6]
