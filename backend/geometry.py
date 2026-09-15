"""Calibrated, distortion-aware geometry. Never invent missing observations."""

from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import least_squares

JOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
BONES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def arrays(camera):
    return tuple(np.asarray(camera[k], dtype=float) for k in ["K", "dist", "R", "T"])


def project(point, camera):
    K, d, R, T = arrays(camera)
    rvec, _ = cv2.Rodrigues(R)
    return cv2.projectPoints(
        np.asarray(point, dtype=float).reshape(-1, 3), rvec, T, K, d
    )[0].reshape(-1, 2)


def camera_center(camera):
    _, _, R, T = arrays(camera)
    return -R.T @ T.reshape(3)


def depth(point, camera):
    _, _, R, T = arrays(camera)
    return float((R @ point + T.reshape(3))[2])


def triangulation_angle(point, cameras):
    rays = [point - camera_center(c) for c in cameras]
    angles = []
    for a, b in combinations(rays, 2):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom > 1e-10:
            angle = np.degrees(np.arccos(np.clip(np.dot(a, b) / denom, -1, 1)))
            angles.append(min(angle, 180 - angle))
    return max(angles, default=0.0)


def dlt(observations, cameras):
    rows = []
    for name, observation in observations.items():
        K, dist, R, T = arrays(cameras[name])
        x, y = cv2.undistortPoints(
            np.array(observation["xy"], dtype=float).reshape(1, 1, 2), K, dist
        ).reshape(2)
        P = np.column_stack((R, T.reshape(3)))
        weight = (
            2.0
            if observation.get("manual")
            else np.sqrt(max(0.05, observation["confidence"]))
        )
        rows.extend([weight * (x * P[2] - P[0]), weight * (y * P[2] - P[1])])
    _, _, vt = np.linalg.svd(rows)
    if abs(vt[-1, 3]) < 1e-10:
        return None
    point = vt[-1, :3] / vt[-1, 3]
    return point if np.isfinite(point).all() else None


def solve_joint(
    observations, cameras, min_confidence=0.35, max_error=6.0, min_angle=1.0
):
    usable = {
        name: o
        for name, o in observations.items()
        if name in cameras
        and o.get("xy") is not None
        and np.isfinite(o["xy"]).all()
        and (o.get("manual") or o["confidence"] >= min_confidence)
    }
    missing = {
        "point": None,
        "status": "missing",
        "error_px": None,
        "cameras": [],
        "excluded": [],
        "angle_deg": None,
        "reason": "Need two usable views",
    }
    if len(usable) < 2:
        return missing
    candidates = []
    for pair in combinations(usable, 2):
        point = dlt({name: usable[name] for name in pair}, cameras)
        if point is None:
            continue
        inliers = [
            name
            for name, o in usable.items()
            if depth(point, cameras[name]) > 0.05
            and np.linalg.norm(project(point, cameras[name])[0] - o["xy"]) <= max_error
        ]
        # Manual labels are authoritative. Conflicting labels must be reviewed.
        if any(o.get("manual") and name not in inliers for name, o in usable.items()):
            continue
        if len(inliers) < 2:
            continue
        angle = triangulation_angle(point, [cameras[name] for name in inliers])
        if angle < min_angle:
            continue
        err = np.mean(
            [
                np.linalg.norm(project(point, cameras[name])[0] - usable[name]["xy"])
                for name in inliers
            ]
        )
        candidates.append((len(inliers), -err, angle, inliers, point))
    if not candidates:
        return {
            **missing,
            "status": "rejected",
            "reason": "Views disagree or triangulation angle is too small",
        }
    support, _, _, inliers, start = max(candidates, key=lambda c: c[:3])
    # Equal-support pairs can each have zero error while predicting different 3D
    # points (e.g. one bad label shifted along an epipolar line). Numerical
    # residual noise cannot establish which pair is correct. Ask for review.
    for other_support, _, _, other_inliers, other_point in candidates:
        if other_support != support or set(other_inliers) == set(inliers):
            continue
        if any(
            np.linalg.norm(
                project(start, cameras[n])[0] - project(other_point, cameras[n])[0]
            )
            > 2 * max_error
            for n in usable
        ):
            return {
                **missing,
                "status": "rejected",
                "reason": "Ambiguous camera agreement: competing view groups produce different 3D points",
            }

    def residual(p):
        return np.concatenate(
            [
                (project(p, cameras[n])[0] - usable[n]["xy"])
                * (
                    2
                    if usable[n].get("manual")
                    else np.sqrt(max(0.05, usable[n]["confidence"]))
                )
                for n in inliers
            ]
        )

    point = least_squares(residual, start, loss="soft_l1", f_scale=2.0, max_nfev=30).x
    errors = [
        float(np.linalg.norm(project(point, cameras[n])[0] - usable[n]["xy"]))
        for n in inliers
    ]
    angle = triangulation_angle(point, [cameras[n] for n in inliers])
    if (
        not np.isfinite(point).all()
        or any(depth(point, cameras[n]) <= 0.05 for n in inliers)
        or max(errors) > max_error
        or angle < min_angle
    ):
        return {
            **missing,
            "status": "rejected",
            "reason": "Refined geometry failed validation",
        }
    return {
        "point": point.tolist(),
        "status": "corrected"
        if any(usable[n].get("manual") for n in inliers)
        else "observed",
        "error_px": float(np.mean(errors)),
        "cameras": inliers,
        "excluded": [n for n in usable if n not in inliers],
        "angle_deg": float(angle),
        "reason": None,
    }


def effective_observations(view):
    result = [dict(o) for o in view["raw"]]
    for index, label in view.get("edits", {}).items():
        result[int(index)] = {
            "xy": label,
            "confidence": 1.0 if label is not None else 0.0,
            "manual": True,
        }
    return result


def reconstruct_frame(frame, calibration, settings):
    cameras = calibration["cameras"]
    effective = {n: effective_observations(v) for n, v in frame["views"].items()}
    joints = [
        solve_joint(
            {n: obs[j] for n, obs in effective.items()},
            cameras,
            settings["min_confidence"],
            settings["max_error_px"],
            settings["min_angle_deg"],
        )
        for j in range(17)
    ]
    frame["pose"] = joints
    for name, view in frame["views"].items():
        view["effective"] = effective[name]
        view["reprojected"] = [
            project(j["point"], cameras[name])[0].tolist()
            if j["point"] is not None
            and depth(np.array(j["point"]), cameras[name]) > 0.05
            else None
            for j in joints
        ]
    frame["quality"] = {
        "valid": sum(j["point"] is not None for j in joints),
        "corrected": sum(j["status"] == "corrected" for j in joints),
        "mean_error_px": float(
            np.mean([j["error_px"] for j in joints if j["error_px"] is not None])
        )
        if any(j["error_px"] is not None for j in joints)
        else None,
        "needs_review": any(j["point"] is None or j["excluded"] for j in joints[5:]),
    }
    return frame
