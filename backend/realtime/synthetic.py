"""Synthetic running motion, camera rig and 2D detector-noise model.

Used for regression tests and the accuracy benchmark. It exercises the
geometry and temporal model with known ground truth; it does not measure the
2D network's accuracy on real people.
"""

from __future__ import annotations

import numpy as np

from .skeletons import COCO17


def look_at(center, target, up=(0, 0, 1)):
    z = np.asarray(target, float) - center
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ center


def ring_calibration(n=4, radius=5.0, height=1.6, size=(1920, 1080), focal=1500.0):
    cams = {}
    for i in range(n):
        a = 2 * np.pi * i / n + 0.3
        c = np.array([radius * np.cos(a), radius * np.sin(a), height])
        R, T = look_at(c, np.array([0, 0, 0.9]))
        cams[f"cam{i + 1}"] = {
            "K": [[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1]],
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
            "R": R.tolist(),
            "T": T.tolist(),
            "image_size": list(size),
        }
    return {
        "format": "mocap_calibration_v1",
        "units": "meters",
        "world_frame": "z_up",
        "quality": {"source": "synthetic", "accepted": True, "stereo_rms_px": 0.0},
        "cameras": cams,
    }


def running_motion(duration=6.0, fps=60.0, speed=3.5, radius=2.0, cadence_hz=1.45):
    """COCO-17 joint trajectories (T,17,3) of a runner circling the origin."""
    t = np.arange(int(duration * fps)) / fps
    theta = speed / radius * t
    fwd = np.stack([-np.sin(theta), np.cos(theta), 0 * t], -1)
    side = np.stack([np.cos(theta), np.sin(theta), 0 * t], -1)  # outward
    up = np.array([0, 0, 1.0])
    phase = 2 * np.pi * cadence_hz * t
    pelvis = np.stack([radius * np.cos(theta), radius * np.sin(theta), 0 * t], -1)
    pelvis[:, 2] = 0.95 + 0.04 * np.cos(2 * phase)
    lean = 0.08
    J = np.zeros((len(t), 17, 3))

    def at(base, f, s, u):
        return base + f[..., None] * fwd + s[..., None] * side + u[..., None] * up

    zero = 0 * t
    neck = at(pelvis, zero + lean * 0.5, zero, zero + 0.5)
    J[:, 11] = at(pelvis, zero, zero - 0.1, zero)  # left hip (inside of circle)
    J[:, 12] = at(pelvis, zero, zero + 0.1, zero)
    J[:, 5] = at(neck, zero, zero - 0.19, zero - 0.02)
    J[:, 6] = at(neck, zero, zero + 0.19, zero - 0.02)
    head = at(neck, zero + 0.06, zero, zero + 0.22)
    J[:, 0] = at(head, zero + 0.09, zero, zero)
    J[:, 1] = at(head, zero + 0.07, zero - 0.03, zero + 0.03)
    J[:, 2] = at(head, zero + 0.07, zero + 0.03, zero + 0.03)
    J[:, 3] = at(head, zero - 0.01, zero - 0.075, zero)
    J[:, 4] = at(head, zero - 0.01, zero + 0.075, zero)
    for hip, knee, ankle, sgn in [(11, 13, 15, 1.0), (12, 14, 16, -1.0)]:
        ph = phase + (0 if sgn > 0 else np.pi)
        thigh = 0.6 * np.sin(ph)  # hip flexion (rad)
        flex = 0.35 + 0.75 * np.clip(np.sin(ph + 1.2), 0, None)  # knee flexion
        J[:, knee] = at(J[:, hip], 0.44 * np.sin(thigh), zero, -0.44 * np.cos(thigh))
        shank = thigh - flex
        J[:, ankle] = at(J[:, knee], 0.43 * np.sin(shank), zero, -0.43 * np.cos(shank))
    for sh, el, wr, sgn in [(5, 7, 9, -1.0), (6, 8, 10, 1.0)]:
        ph = phase + (0 if sgn > 0 else np.pi)
        upper = 0.5 * np.sin(ph)
        J[:, el] = at(J[:, sh], 0.29 * np.sin(upper), zero, -0.29 * np.cos(upper))
        fore = upper + 1.4
        J[:, wr] = at(J[:, el], 0.26 * np.sin(fore), zero, -0.26 * np.cos(fore))
    return t, J


def detector_noise(
    uv,
    rng,
    sigma_px=2.5,
    outlier_rate=0.03,
    dropout=0.03,
    swap_rate=0.01,
    swap_len=6,
):
    """Apply a pessimistic 2D detector error model to clean projections.

    uv (T,C,J,2). Returns noisy uv, per-point sigma and confidence.
    """
    T, C, J, _ = uv.shape
    noisy = uv + rng.normal(0, sigma_px, uv.shape)
    out = rng.random((T, C, J)) < outlier_rate
    mag = rng.uniform(20, 90, (T, C, J))
    ang = rng.uniform(0, 2 * np.pi, (T, C, J))
    noisy[..., 0] += np.where(out, mag * np.cos(ang), 0)
    noisy[..., 1] += np.where(out, mag * np.sin(ang), 0)
    conf = np.clip(rng.normal(0.85, 0.08, (T, C, J)), 0.3, 1.0)
    conf = np.where(out, np.clip(conf - 0.25, 0.2, 1), conf)
    drop = rng.random((T, C, J)) < dropout
    noisy[drop] = np.nan
    # Bursts of left/right leg confusion in one camera.
    legs = [(11, 12), (13, 14), (15, 16)]
    t = 0
    while t < T:
        if rng.random() < swap_rate:
            c = rng.integers(C)
            for k in range(t, min(T, t + swap_len)):
                for a, b in legs:
                    noisy[k, c, [a, b]] = noisy[k, c, [b, a]]
            t += swap_len
        t += 1
    return noisy, conf


SKELETON = COCO17
