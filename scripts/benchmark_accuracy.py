"""Accuracy benchmark of the 3D + temporal pipeline on synthetic running.

Compares the MVP per-frame solver (backend.geometry) with the real-time
triangulator, left/right repair, live Kalman filter, saved RTS smoothing and
the bone-length constraint. Ground truth is known exactly; 2D noise follows a
pessimistic detector model (Gaussian noise, outliers, dropouts, L/R swaps).

    python scripts/benchmark_accuracy.py --cams 4 --fps 60
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.realtime.skeletons import COCO17  # noqa: E402
from backend.realtime.synthetic import (  # noqa: E402
    detector_noise,
    ring_calibration,
    running_motion,
)
from backend.realtime.temporal import (  # noqa: E402
    BoneLengths,
    Kalman3D,
    default_jerk,
    smooth_sequence,
)
from backend.realtime.triangulation import (  # noqa: E402
    CameraRig,
    repair_left_right,
    triangulate,
)


def stats(est, gt):
    ok = np.isfinite(est).all(-1)
    e = np.linalg.norm(est - gt, axis=-1)[ok] * 1000
    # Jitter: RMS of the second difference of the error (frame-to-frame shake).
    d = est - gt
    acc = d[2:] - 2 * d[1:-1] + d[:-2]
    acc_ok = np.isfinite(acc).all(-1)
    jitter = (
        np.sqrt(np.mean(np.sum(acc[acc_ok] ** 2, -1))) * 1000
        if acc_ok.any()
        else np.nan
    )
    return {
        "coverage": ok.mean() * 100,
        "mpjpe": e.mean(),
        "p95": np.percentile(e, 95),
        "jitter": jitter,
    }


def run(cams=4, fps=60.0, duration=6.0, seed=0, old=True, sigma_px=2.5):
    rng = np.random.default_rng(seed)
    cal = ring_calibration(cams)
    rig = CameraRig.from_calibration(cal)
    times, gt = running_motion(duration, fps)
    T, J = gt.shape[:2]
    clean = np.stack([rig.project(gt[t])[0] for t in range(T)])  # (T,C,J,2)
    uv, conf = detector_noise(clean, rng, sigma_px=sigma_px)
    sigma = sigma_px / conf
    results = {}

    if old:
        from backend.geometry import solve_joint

        n = min(T, int(fps))  # 1 s: the per-joint loop is slow
        est = np.full((n, J, 3), np.nan)
        start = time.perf_counter()
        for t in range(n):
            for j in range(J):
                obs = {
                    name: {
                        "xy": None
                        if np.isnan(uv[t, c, j]).any()
                        else uv[t, c, j].tolist(),
                        "confidence": float(conf[t, c, j]),
                    }
                    for c, name in enumerate(rig.names)
                }
                r = solve_joint(obs, cal["cameras"])
                if r["point"] is not None:
                    est[t, j] = r["point"]
        ms = (time.perf_counter() - start) / n * 1000
        results["MVP per-frame solver (1 s sample)"] = stats(est, gt[:n]) | {"ms": ms}

    raw = np.full((T, J, 3), np.nan)
    rawfix = np.full((T, J, 3), np.nan)
    live = np.full((T, J, 3), np.nan)
    Z = np.full((T, J, 3), np.nan)
    Rs = np.tile(np.eye(3), (T, J, 1, 1))
    valid = np.zeros((T, J), bool)
    kf = Kalman3D(J, default_jerk(COCO17))
    tri_time = 0.0
    for t in range(T):
        start = time.perf_counter()
        raw[t] = triangulate(rig, uv[t], sigma[t])["X"]
        kf.predict(times[t])
        pred, _ = kf.state()
        u, s = uv[t], sigma[t]
        if np.isfinite(pred).any():
            pred_uv, _ = rig.project(pred)
            u, s, _ = repair_left_right(u, s, pred_uv, COCO17.swap_groups)
        res = triangulate(rig, u, s)
        tri_time += time.perf_counter() - start
        rawfix[t] = res["X"]
        ok = np.isfinite(res["X"]).all(-1)
        Z[t][ok], Rs[t][ok], valid[t] = res["X"][ok], res["cov"][ok], ok
        kf.update(res["X"], np.where(ok[:, None, None], res["cov"], 0), ok)
        live[t] = kf.state()[0]
    results["Real-time triangulation"] = stats(raw, gt) | {
        "ms": tri_time / T * 1000 / 2
    }
    results["+ left/right repair"] = stats(rawfix, gt)
    results["+ live Kalman filter (display)"] = stats(live, gt)
    _, smoothed, sig = smooth_sequence(times, Z, Rs, valid, default_jerk(COCO17))
    results["+ RTS smoothing (saved result)"] = stats(smoothed, gt)
    bones = BoneLengths(COCO17)
    lengths = bones.fit_sequence(smoothed, sig)
    constrained = bones.apply(smoothed, lengths)
    results["+ bone-length constraint (saved)"] = stats(constrained, gt)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cams", type=int, default=4)
    p.add_argument("--fps", type=float, default=60)
    p.add_argument("--duration", type=float, default=6)
    p.add_argument("--sigma", type=float, default=2.5)
    p.add_argument("--no-old", action="store_true")
    a = p.parse_args()
    res = run(a.cams, a.fps, a.duration, old=not a.no_old, sigma_px=a.sigma)
    print(
        f"{a.cams} cameras, {a.fps:g} fps, 2D noise σ={a.sigma}px + outliers/dropouts/L-R swaps"
    )
    print(
        f"{'method':38s} {'cover%':>7s} {'MPJPE':>7s} {'p95':>7s} {'jitter':>7s} {'ms/frame':>9s}"
    )
    for k, v in res.items():
        ms = f"{v['ms']:.2f}" if "ms" in v else ""
        print(
            f"{k:38s} {v['coverage']:7.1f} {v['mpjpe']:6.1f}mm {v['p95']:6.1f}mm {v['jitter']:6.1f}mm {ms:>9s}"
        )


if __name__ == "__main__":
    main()
