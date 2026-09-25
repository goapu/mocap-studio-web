"""Vectorised, uncertainty-aware multi-view triangulation.

All joints of a frame are solved together:
1. weighted DLT on undistorted rays (weights = focal / 2D sigma),
2. Gauss–Newton refinement of the pixel reprojection error,
3. robust view selection – the worst view is dropped and the joint re-solved
   (instead of rejecting the whole joint when one camera disagrees),
4. a 3×3 covariance from the Gauss–Newton normal matrix, which captures
   depth uncertainty caused by narrow triangulation angles.

A frame with 6 cameras × 26 joints solves in well under a millisecond on a
laptop CPU, compared to ~1.6 s for the per-joint loop in ``backend.geometry``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MISSING, OBSERVED, CONFLICT = 0, 1, 2


@dataclass
class CameraRig:
    names: list[str]
    K: np.ndarray  # (C,3,3)
    dist: list[np.ndarray]
    R: np.ndarray  # (C,3,3)
    T: np.ndarray  # (C,3)
    image_size: np.ndarray  # (C,2) width,height
    world_frame: str = "camera"

    def __post_init__(self):
        self.rvecs = [cv2.Rodrigues(r)[0] for r in self.R]
        self.undistort_needed = [bool(np.any(np.abs(d) > 0)) for d in self.dist]

    @classmethod
    def from_calibration(cls, calibration: dict, names=None, video_sizes=None):
        """Build a rig; intrinsics are rescaled if a video is a uniformly
        resized copy of the calibrated resolution (same aspect ratio)."""
        cams = calibration["cameras"]
        names = list(names or cams)
        K, dist, R, T, sizes = [], [], [], [], []
        for name in names:
            cam = cams[name]
            k = np.array(cam["K"], float)
            size = np.array(cam["image_size"], float)
            if video_sizes and name in video_sizes:
                vw, vh = video_sizes[name]
                sx, sy = vw / size[0], vh / size[1]
                if abs(sx - sy) > 1e-3:
                    raise ValueError(
                        f"{name}: video is {vw}×{vh} but calibration is "
                        f"{int(size[0])}×{int(size[1])} (different aspect ratio). "
                        "Use footage at the calibrated resolution."
                    )
                if abs(sx - 1) > 1e-6:
                    k[0, :] *= sx
                    k[1, :] *= sy
                    size = np.array([vw, vh], float)
            K.append(k)
            dist.append(np.array(cam.get("dist", []), float).reshape(-1))
            R.append(np.array(cam["R"], float))
            T.append(np.array(cam["T"], float).reshape(3))
            sizes.append(size)
        return cls(
            names,
            np.array(K),
            dist,
            np.array(R),
            np.array(T),
            np.array(sizes),
            calibration.get("world_frame", "camera"),
        )

    @property
    def count(self) -> int:
        return len(self.names)

    @property
    def focal(self) -> np.ndarray:
        return 0.5 * (self.K[:, 0, 0] + self.K[:, 1, 1])

    def centers(self) -> np.ndarray:
        return -np.einsum("cji,cj->ci", self.R, self.T)

    def undistort(self, uv: np.ndarray) -> np.ndarray:
        """(C,J,2) pixels → (C,J,2) normalised image coordinates (NaN kept)."""
        out = np.full(uv.shape, np.nan)
        for c in range(self.count):
            pts = uv[c]
            ok = np.isfinite(pts).all(-1)
            if ok.any() and not self.undistort_needed[c]:
                k = self.K[c]
                p = pts[ok] - k[:2, 2]
                out[c, ok] = np.linalg.solve(k[:2, :2], p.T).T
            elif ok.any():
                und = cv2.undistortPointsIter(
                    pts[ok].reshape(-1, 1, 2).astype(np.float64),
                    self.K[c],
                    self.dist[c],
                    None,
                    None,
                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-8),
                )
                out[c, ok] = und.reshape(-1, 2)
        return out

    def project(self, X: np.ndarray):
        """(J,3) world points → pixels (C,J,2) and camera depth (C,J)."""
        J = len(X)
        uv = np.full((self.count, J, 2), np.nan)
        depth = np.full((self.count, J), np.nan)
        ok = np.isfinite(X).all(-1)
        if not ok.any():
            return uv, depth
        pts = X[ok].astype(np.float64)
        cam = np.einsum("cij,pj->cpi", self.R, pts) + self.T[:, None, :]
        depth[:, ok] = cam[..., 2]
        for c in range(self.count):
            if self.undistort_needed[c]:
                proj, _ = cv2.projectPoints(
                    pts.reshape(-1, 1, 3),
                    self.rvecs[c],
                    self.T[c],
                    self.K[c],
                    self.dist[c],
                )
                uv[c, ok] = proj.reshape(-1, 2)
            else:
                z = np.where(np.abs(cam[c, :, 2]) < 1e-9, 1e-9, cam[c, :, 2])
                xy = cam[c, :, :2] / z[:, None]
                uv[c, ok] = xy @ self.K[c][:2, :2].T + self.K[c][:2, 2]
        return uv, depth


def _normal_system(X, rig, norm, weight, used):
    """Residuals (in sigma units) and Jacobians for Gauss–Newton."""
    Xc = np.einsum("cij,pj->cpi", rig.R, X) + rig.T[:, None, :]  # (C,J,3)
    z = Xc[..., 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    proj = Xc[..., :2] / zs[..., None]
    r = (proj - norm) * weight[..., None]  # (C,J,2)
    # d(x/z)/dX = (R0 - x R2)/z ; d(y/z)/dX = (R1 - y R2)/z
    R0 = rig.R[:, None, 0, :]
    R1 = rig.R[:, None, 1, :]
    R2 = rig.R[:, None, 2, :]
    jx = (R0 - proj[..., 0:1] * R2) / zs[..., None]
    jy = (R1 - proj[..., 1:2] * R2) / zs[..., None]
    Jm = np.stack([jx, jy], -2) * weight[..., None, None]  # (C,J,2,3)
    m = used[..., None]
    r = np.where(m, r, 0.0)
    Jm = np.where(m[..., None], Jm, 0.0)
    H = np.einsum("cpki,cpkj->pij", Jm, Jm)
    g = np.einsum("cpki,cpk->pi", Jm, r)
    return H, g, z


def triangulate(
    rig: CameraRig,
    uv: np.ndarray,
    sigma: np.ndarray,
    gate_px: float = 8.0,
    gate_sigma: float = 4.0,
    iterations: int = 3,
):
    """Solve all joints of one frame.

    uv: (C,J,2) pixel observations (NaN = missing); sigma: (C,J) 2D standard
    deviation in pixels. Returns dict with X (J,3), cov (J,3,3), used (C,J),
    err_px (C,J), status (J,).
    """
    C, J = uv.shape[:2]
    valid = np.isfinite(uv).all(-1) & np.isfinite(sigma) & (sigma > 0)
    norm = rig.undistort(uv)
    valid &= np.isfinite(norm).all(-1)
    weight = np.where(valid, rig.focal[:, None] / np.where(valid, sigma, 1.0), 0.0)
    used = valid.copy()
    X = np.full((J, 3), np.nan)
    cov = np.full((J, 3, 3), np.nan)
    err = np.full((C, J), np.nan)
    status = np.full(J, MISSING, int)
    P = np.concatenate([rig.R, rig.T[:, :, None]], 2)  # (C,3,4)
    active = used.sum(0) >= 2
    for _ in range(max(C - 1, 1)):
        if not active.any():
            break
        idx = np.where(active)[0]
        n = np.nan_to_num(norm[:, idx])
        w = weight[:, idx] * used[:, idx]
        rows_x = n[..., 0:1] * P[:, None, 2, :] - P[:, None, 0, :]
        rows_y = n[..., 1:2] * P[:, None, 2, :] - P[:, None, 1, :]
        A = np.concatenate([rows_x, rows_y], 0) * np.concatenate([w, w], 0)[..., None]
        A = A.transpose(1, 0, 2)  # (j, 2C, 4)
        _, _, vt = np.linalg.svd(A)
        h = vt[:, -1, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            Xi = h[:, :3] / h[:, 3:4]
        for _ in range(iterations):
            H, g, _ = _normal_system(
                Xi, rig, np.nan_to_num(norm[:, idx]), weight[:, idx], used[:, idx]
            )
            H = H + np.eye(3) * 1e-9
            try:
                step = np.linalg.solve(H, -g[..., None])[..., 0]
            except np.linalg.LinAlgError:
                break
            Xi = Xi + np.where(np.isfinite(step), step, 0.0)
        X[idx] = Xi
        proj, depth = rig.project(Xi)
        e = np.linalg.norm(proj - uv[:, idx], axis=-1)
        e = np.where(depth > 0.05, e, np.inf)
        err[:, idx] = np.where(valid[:, idx], e, np.nan)
        thr = np.maximum(
            gate_px, gate_sigma * np.where(valid[:, idx], sigma[:, idx], 1)
        )
        bad = used[:, idx] & (e > thr)
        ratio = np.where(bad, e / thr, -1.0)
        worst = ratio.argmax(0)
        has_bad = bad.any(0)
        can_drop = has_bad & (used[:, idx].sum(0) > 2)
        if not can_drop.any():
            break
        cols = idx[can_drop]
        used[worst[can_drop], cols] = False
        active = np.zeros(J, bool)
        active[cols] = True
    # Final covariance / status on the accepted view sets.
    solvable = used.sum(0) >= 2
    if solvable.any():
        idx = np.where(solvable)[0]
        H, _, z = _normal_system(
            X[idx], rig, np.nan_to_num(norm[:, idx]), weight[:, idx], used[:, idx]
        )
        try:
            cov[idx] = np.linalg.inv(H + np.eye(3) * 1e-12)
        except np.linalg.LinAlgError:
            pass
        e = err[:, idx]
        thr = np.maximum(
            gate_px, gate_sigma * np.where(valid[:, idx], sigma[:, idx], 1)
        )
        inconsistent = (used[:, idx] & ~(e <= thr)).any(0)
        front = np.all(np.where(used[:, idx], z > 0.05, True), 0)
        good = np.isfinite(X[idx]).all(-1) & front & np.isfinite(cov[idx]).all((1, 2))
        status[idx] = np.where(
            good, np.where(inconsistent, CONFLICT, OBSERVED), MISSING
        )
        # Conflicting 2-view solutions stay available but with inflated
        # uncertainty, so the temporal filter can down-weight or reject them.
        scale = np.nanmax(np.where(used[:, idx], e / thr, 0), 0) ** 2
        cov[idx] *= np.where(inconsistent, np.maximum(scale, 1) * 25, 1)[:, None, None]
    X[status == MISSING] = np.nan
    cov[status == MISSING] = np.nan
    used &= (status != MISSING)[None, :]
    return {"X": X, "cov": cov, "used": used, "err_px": err, "status": status}


def repair_left_right(uv, sigma, predicted_uv, swap_groups, ratio=0.6, min_px=6.0):
    """Undo per-camera left/right confusions (common for legs while running).

    For each camera and each joint group (e.g. hips+knees+ankles), compare the
    detection against the projection of the temporally predicted 3D pose with
    and without swapping sides. A swap is applied only when it explains the
    image clearly better. Returns (uv, sigma, swapped[C, groups]).
    """
    uv = uv.copy()
    sigma = sigma.copy()
    C = uv.shape[0]
    swapped = np.zeros((C, len(swap_groups)), bool)
    for c in range(C):
        for g, pairs in enumerate(swap_groups):
            a = np.array([p[0] for p in pairs])
            b = np.array([p[1] for p in pairs])
            q_a, q_b = predicted_uv[c, a], predicted_uv[c, b]
            d_a, d_b = uv[c, a], uv[c, b]
            ok = (
                np.isfinite(q_a).all(-1)
                & np.isfinite(q_b).all(-1)
                & np.isfinite(d_a).all(-1)
                & np.isfinite(d_b).all(-1)
            )
            if ok.sum() < 2:
                continue
            keep = np.linalg.norm(d_a - q_a, axis=-1) + np.linalg.norm(
                d_b - q_b, axis=-1
            )
            swap = np.linalg.norm(d_a - q_b, axis=-1) + np.linalg.norm(
                d_b - q_a, axis=-1
            )
            keep, swap = keep[ok].sum(), swap[ok].sum()
            if keep > min_px * ok.sum() and swap < ratio * keep:
                uv[c, a], uv[c, b] = d_b.copy(), d_a.copy()
                sigma[c, a], sigma[c, b] = sigma[c, b].copy(), sigma[c, a].copy()
                swapped[c, g] = True
    return uv, sigma, swapped
