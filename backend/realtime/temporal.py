"""Temporal models: 2D One-Euro filter, 3D Kalman filter + RTS smoother and a
rigid-limb (bone length) constraint.

Live display uses the causal Kalman filter (no look-ahead, no added latency).
Saved results use the Rauch–Tung–Striebel smoother over the whole recording –
every frame is informed by past *and* future frames, which is where processing
every frame (and higher capture frame rates) pays off in accuracy.
"""

from __future__ import annotations

import math

import numpy as np

from .skeletons import Skeleton


class OneEuro:
    """Vectorised One-Euro filter (Casiez et al. 2012) for 2D keypoints."""

    def __init__(self, min_cutoff=1.2, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = None
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x = self.dx = self.t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, float)
        if self.x is None or self.t is None or t <= self.t:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self.t
        prev = np.where(np.isfinite(self.x), self.x, x)
        dx = (x - prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = np.where(np.isfinite(dx), a_d * dx + (1 - a_d) * self.dx, self.dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)
        out = np.where(np.isfinite(x), a * x + (1 - a) * prev, np.nan)
        self.x = np.where(np.isfinite(out), out, self.x)
        self.t = t
        return out


def _transition(dt):
    F1 = np.array([[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]])
    Q1 = np.array(
        [
            [dt**5 / 20, dt**4 / 8, dt**3 / 6],
            [dt**4 / 8, dt**3 / 3, dt**2 / 2],
            [dt**3 / 6, dt**2 / 2, dt],
        ]
    )
    I3 = np.eye(3)
    return np.kron(F1, I3), np.kron(Q1, I3)


H = np.hstack([np.eye(3), np.zeros((3, 6))])


def default_jerk(skeleton: Skeleton) -> np.ndarray:
    """Jerk spectral density per joint (m²/s⁵). Extremities move more abruptly."""
    q = np.full(skeleton.size, 2000.0)
    for i, name in enumerate(skeleton.joints):
        if any(k in name for k in ("wrist", "ankle", "toe", "heel")):
            q[i] = 8000.0
        elif any(k in name for k in ("elbow", "knee")):
            q[i] = 4000.0
    return q


class Kalman3D:
    """Constant-acceleration Kalman filter for every joint (batched).

    Measurements carry their own 3×3 covariance from triangulation, so a
    joint seen by two cameras at a narrow angle is trusted less along depth.
    Innovations beyond the chi-square gate are rejected as outliers; a run of
    rejections re-initialises the joint (e.g. after a genuine fast movement).
    """

    GATE = 25.0  # chi-square, 3 dof, ~99.998 %

    def __init__(
        self,
        joints: int,
        jerk: np.ndarray,
        max_gap_s: float = 0.3,
        reset_after: int = 4,
        min_sigma_m: float = 0.004,
    ):
        self.J = joints
        self.q = np.asarray(jerk, float)
        self.max_gap_s = max_gap_s
        self.reset_after = reset_after
        self.min_var = min_sigma_m**2
        self.x = np.zeros((joints, 9))
        self.P = np.tile(np.eye(9), (joints, 1, 1))
        self.alive = np.zeros(joints, bool)
        self.since = np.full(joints, np.inf)  # seconds since last update
        self.rejects = np.zeros(joints, int)
        self.t = None

    def _init(self, j, z, R):
        self.x[j] = 0
        self.x[j, :3] = z
        P = np.zeros((9, 9))
        P[:3, :3] = R
        P[3:6, 3:6] = np.eye(3) * 4.0  # (2 m/s)^2
        P[6:, 6:] = np.eye(3) * 400.0  # (20 m/s²)^2
        self.P[j] = P
        self.alive[j] = True
        self.since[j] = 0
        self.rejects[j] = 0

    def predict(self, t: float):
        if self.t is None:
            self.t = t
            return
        dt = max(t - self.t, 1e-4)
        self.t = t
        F, Q1 = _transition(dt)
        self.x = self.x @ F.T
        self.P = F @ self.P @ F.T + self.q[:, None, None] * Q1
        self.since += dt
        self.alive &= self.since <= self.max_gap_s

    def update(self, z: np.ndarray, R: np.ndarray, valid: np.ndarray):
        """Returns per-joint flags: 0 none, 1 updated, 2 rejected, 3 (re)initialised."""
        flags = np.zeros(self.J, int)
        R = R + np.eye(3) * self.min_var
        for j in np.where(valid)[0]:
            if not self.alive[j]:
                self._init(j, z[j], R[j])
                flags[j] = 3
                continue
            y = z[j] - self.x[j, :3]
            S = self.P[j, :3, :3] + R[j]
            try:
                Si = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                continue
            d2 = float(y @ Si @ y)
            if d2 > self.GATE:
                self.rejects[j] += 1
                if self.rejects[j] >= self.reset_after:
                    self._init(j, z[j], R[j])
                    flags[j] = 3
                else:
                    flags[j] = 2
                continue
            K = self.P[j] @ H.T @ Si
            self.x[j] = self.x[j] + K @ y
            IKH = np.eye(9) - K @ H
            self.P[j] = IKH @ self.P[j] @ IKH.T + K @ R[j] @ K.T
            self.since[j] = 0
            self.rejects[j] = 0
            flags[j] = 1
        return flags

    def state(self):
        pos = np.where(self.alive[:, None], self.x[:, :3], np.nan)
        var = np.where(self.alive[:, None], self.P[:, [0, 1, 2], [0, 1, 2]], np.nan)
        return pos, np.sqrt(var)


def smooth_sequence(times, Z, Rs, valid, jerk, max_gap_s=0.3, reset_after=4):
    """Forward Kalman + backward RTS smoothing for a whole recording.

    times (T,), Z (T,J,3), Rs (T,J,3,3), valid (T,J).
    Returns filtered (T,J,3), smoothed (T,J,3), smoothed sigma (T,J,3).
    """
    T, J = Z.shape[:2]
    kf = Kalman3D(J, jerk, max_gap_s, reset_after)
    xp = np.zeros((T, J, 9))
    Pp = np.zeros((T, J, 9, 9))
    xf = np.zeros((T, J, 9))
    Pf = np.zeros((T, J, 9, 9))
    alive = np.zeros((T, J), bool)
    restart = np.zeros((T, J), bool)
    Fs = []
    for t in range(T):
        was_alive = kf.alive.copy()
        kf.predict(times[t])
        dt = times[t] - times[t - 1] if t else 1.0
        Fs.append(_transition(max(dt, 1e-4))[0])
        xp[t], Pp[t] = kf.x, kf.P
        flags = kf.update(Z[t], Rs[t], valid[t])
        restart[t] = (flags == 3) | (~was_alive & kf.alive)
        xf[t], Pf[t] = kf.x, kf.P
        alive[t] = kf.alive
    xs, Ps = xf.copy(), Pf.copy()
    for t in range(T - 2, -1, -1):
        F = Fs[t + 1]
        link = alive[t] & alive[t + 1] & ~restart[t + 1]
        if not link.any():
            continue
        j = np.where(link)[0]
        try:
            G = Pf[t, j] @ F.T @ np.linalg.inv(Pp[t + 1, j])
        except np.linalg.LinAlgError:
            continue
        xs[t, j] = xf[t, j] + np.einsum("jab,jb->ja", G, xs[t + 1, j] - xp[t + 1, j])
        Ps[t, j] = Pf[t, j] + G @ (Ps[t + 1, j] - Pp[t + 1, j]) @ G.transpose(0, 2, 1)
    nan = ~alive[..., None]
    filtered = np.where(nan, np.nan, xf[..., :3])
    smoothed = np.where(nan, np.nan, xs[..., :3])
    sig = np.sqrt(np.clip(Ps[..., [0, 1, 2], [0, 1, 2]], 0, None))
    return filtered, smoothed, np.where(nan, np.nan, sig)


class BoneLengths:
    """Robust subject-specific limb lengths and a rigid-limb projection."""

    def __init__(self, skeleton: Skeleton, history=900, min_samples=15):
        self.skeleton = skeleton
        self.edges = list(skeleton.limb_tree)
        self.desc = skeleton.descendants()
        self.history = history
        self.min_samples = min_samples
        self.samples: list[list[float]] = [[] for _ in self.edges]

    def observe(self, X: np.ndarray, sigma: np.ndarray, max_sigma_m=0.02):
        for k, (a, b) in enumerate(self.edges):
            if np.isfinite(X[[a, b]]).all() and np.nanmax(sigma[[a, b]]) < max_sigma_m:
                buf = self.samples[k]
                buf.append(float(np.linalg.norm(X[b] - X[a])))
                if len(buf) > self.history:
                    del buf[: len(buf) - self.history]

    def lengths(self) -> np.ndarray:
        return np.array(
            [
                np.median(s) if len(s) >= self.min_samples else np.nan
                for s in self.samples
            ]
        )

    def fit_sequence(self, X: np.ndarray, sigma: np.ndarray, max_sigma_m=0.02):
        for t in range(len(X)):
            self.observe(X[t], sigma[t], max_sigma_m)
        return self.lengths()

    def apply(self, X: np.ndarray, lengths=None) -> np.ndarray:
        """Project limbs to target lengths. X (J,3) or (T,J,3)."""
        lengths = self.lengths() if lengths is None else lengths
        out = np.array(X, float, copy=True)
        single = out.ndim == 2
        if single:
            out = out[None]
        for k, (a, b) in enumerate(self.edges):
            L = lengths[k]
            if not np.isfinite(L):
                continue
            vec = out[:, b] - out[:, a]
            cur = np.linalg.norm(vec, axis=-1)
            ok = np.isfinite(cur) & (cur > 1e-6)
            if not ok.any():
                continue
            delta = np.zeros_like(vec)
            delta[ok] = vec[ok] * (L / cur[ok] - 1)[:, None]
            for node in self.desc[b]:
                out[:, node] += delta
        return out[0] if single else out
