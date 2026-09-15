"""Versioned calibration IO and quality-gated paired chessboard calibration."""

import io
import json
import zipfile
from itertools import combinations

import cv2
import numpy as np

FORMAT = "mocap_calibration_v1"
CAMERA_NAMES = {f"cam{i}" for i in range(1, 7)}
MAX_RMS_PX = 1.5
MAX_VIEW_RMS_PX = 3.0
MIN_BOARD_NORMAL_SPAN_DEG = 8.0


def _finite_array(value, shape, label, flatten=False):
    try:
        result = np.asarray(value, dtype=float)
        if flatten and result.size == np.prod(shape):
            result = result.reshape(shape)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}: expected finite numeric values.") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label}: expected finite values with shape {shape}.")
    return result


def _rms(value, label, limit=MAX_RMS_PX):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label}: RMS must be a finite nonnegative number.")
    if value > limit:
        raise ValueError(f"Calibration quality gate: {label} exceeds {limit:g} px.")
    return float(value)


def validate(calibration):
    if not isinstance(calibration, dict):
        raise ValueError("Calibration must be a JSON object.")
    if calibration.get("format", FORMAT) != FORMAT:
        raise ValueError(f"Unsupported calibration format; expected {FORMAT}.")
    if calibration.get("units") != "meters":
        raise ValueError("Calibration units must be meters.")
    cameras = calibration.get("cameras", {})
    if not isinstance(cameras, dict) or not 2 <= len(cameras) <= 6:
        raise ValueError("Provide calibration for 2–6 cameras as a cameras object.")
    for name, camera in cameras.items():
        if name not in CAMERA_NAMES:
            raise ValueError("Camera names must be cam1 … cam6.")
        if not isinstance(camera, dict):
            raise ValueError(f"{name}: camera parameters must be an object.")
        for key, shape in [("K", (3, 3)), ("R", (3, 3)), ("T", (3,))]:
            value = _finite_array(
                camera.get(key), shape, f"{name}: invalid {key}", key == "T"
            )
            camera[key] = value.tolist()
        K, R = np.array(camera["K"]), np.array(camera["R"])
        if (
            K[0, 0] <= 0
            or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose([K[0, 1], K[1, 0]], 0, atol=1e-10, rtol=0)
        ):
            # OpenCV projectPoints/undistortPoints use fx, fy, cx, cy and ignore skew.
            raise ValueError(
                f"{name}: intrinsic matrix requires positive focal lengths and zero skew."
            )
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(R), 1, atol=1e-5
        ):
            raise ValueError(f"{name}: rotation must be orthonormal and right handed.")
        try:
            dist = np.asarray(camera.get("dist", []), dtype=float).reshape(-1)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name}: invalid distortion coefficients.") from exc
        if len(dist) not in [4, 5, 8, 12, 14] or not np.isfinite(dist).all():
            raise ValueError(f"{name}: invalid distortion coefficients.")
        camera["dist"] = dist.tolist()
        size = camera.get("image_size", [])
        if (
            not isinstance(size, (list, tuple, np.ndarray))
            or len(size) != 2
            or any(
                isinstance(x, (bool, np.bool_))
                or not isinstance(x, (int, np.integer))
                or x <= 0
                for x in size
            )
        ):
            raise ValueError(f"{name}: image_size [width, height] is required.")
        camera["image_size"] = [int(x) for x in size]
        if "intrinsic_rms_px" in camera:
            _rms(camera["intrinsic_rms_px"], f"{name} intrinsic RMS")
    centers = [-np.array(c["R"]).T @ np.array(c["T"]) for c in cameras.values()]
    if max(np.linalg.norm(a - b) for a, b in combinations(centers, 2)) < 0.01:
        raise ValueError(
            "Camera baseline is less than 1 cm or cameras share a position."
        )
    quality = calibration.get("quality", {})
    if not isinstance(quality, dict):
        raise ValueError("Calibration quality must be an object.")
    if quality.get("accepted") is not None and not isinstance(
        quality["accepted"], bool
    ):
        raise ValueError("Calibration quality.accepted must be true, false, or null.")
    if quality.get("accepted") is False:
        raise ValueError("This calibration failed its quality gate.")
    if "stereo_rms_px" in quality:
        _rms(quality["stereo_rms_px"], "Stereo calibration RMS")
    if calibration.get("world_frame") not in ["camera", "z_up"]:
        raise ValueError(
            "world_frame must be camera (x right, y down, z forward) or z_up."
        )
    calibration["format"] = FORMAT
    return calibration


def read_calibration(content, filename):
    try:
        return _read_calibration(content, filename)
    except KeyError as exc:
        raise ValueError(
            f"Incomplete NPZ calibration: missing {exc}. Use the calibration builder or supply a complete JSON."
        ) from exc
    except (TypeError, AttributeError, EOFError, OSError, zipfile.BadZipFile) as exc:
        raise ValueError(
            "Invalid calibration file. Supply a complete calibration JSON or NPZ."
        ) from exc


def _read_calibration(content, filename):
    if filename.lower().endswith(".json"):
        return validate(json.loads(content))
    if not filename.lower().endswith(".npz"):
        raise ValueError("Upload a calibration JSON or NPZ file.")
    archive = np.load(io.BytesIO(content), allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("Expected an NPZ calibration archive.")
    with archive as z:
        if "units" in z and z["units"].tolist() != "meters":
            raise ValueError("Calibration units must be meters.")
        if "camera_names" in z:
            names = z["camera_names"]
            if names.ndim != 1 or not 2 <= len(names) <= 6:
                raise ValueError("NPZ camera_names must list 2–6 unique cameras.")
            names = names.tolist()
            if any(
                not isinstance(n, str) or n not in CAMERA_NAMES for n in names
            ) or len(set(names)) != len(names):
                raise ValueError("NPZ camera_names must list unique names cam1 … cam6.")
            cams = {
                name: {
                    k: z[f"{name}_{k}"].tolist()
                    for k in ["K", "dist", "R", "T", "image_size"]
                }
                for name in names
            }
            if all(k in z for k in ["R", "T"]) and "cam1" in cams and "cam2" in cams:
                R1 = _finite_array(cams["cam1"]["R"], (3, 3), "cam1 R")
                R2 = _finite_array(cams["cam2"]["R"], (3, 3), "cam2 R")
                T1 = _finite_array(cams["cam1"]["T"], (3,), "cam1 T", True)
                T2 = _finite_array(cams["cam2"]["T"], (3,), "cam2 T", True)
                legacy_R = _finite_array(z["R"], (3, 3), "Legacy R")
                legacy_T = _finite_array(z["T"], (3,), "Legacy T", True)
                rel = R2 @ R1.T
                if not np.allclose(rel, legacy_R, atol=1e-5) or not np.allclose(
                    T2 - rel @ T1, legacy_T, atol=1e-5
                ):
                    raise ValueError(
                        "NPZ contains conflicting legacy and named-camera geometry. Recalibrate or supply a consistent JSON."
                    )
        else:
            if "image_size" not in z:
                raise ValueError(
                    "Legacy NPZ lacks image_size. Use the calibration builder or add explicit sizes to JSON."
                )
            cams = {
                f"cam{i}": {
                    "K": z[f"K{i}"].tolist(),
                    "dist": z[f"dist{i}"].tolist(),
                    "R": (np.eye(3) if i == 1 else z["R"]).tolist(),
                    "T": (np.zeros(3) if i == 1 else z["T"]).tolist(),
                    "image_size": z["image_size"].tolist(),
                }
                for i in [1, 2]
            }
        first = cams.get("cam1", next(iter(cams.values())))
        first_R = _finite_array(first["R"], (3, 3), "Reference R")
        first_T = _finite_array(first["T"], (3,), "Reference T", True)
        if not np.allclose(first_R, np.eye(3)) or not np.allclose(first_T, 0):
            raise ValueError(
                "NPZ world orientation is ambiguous. Supply JSON with an explicit camera or z_up world_frame."
            )
        quality = {"source": "imported", "accepted": None}
        if "stereo_rms" in z:
            rms = np.asarray(z["stereo_rms"])
            if rms.size != 1:
                raise ValueError("NPZ stereo_rms must be a scalar.")
            rms = _rms(float(rms.reshape(-1)[0]), "Stereo calibration RMS")
            quality.update(stereo_rms_px=rms, accepted=True)
    return validate(
        {
            "units": "meters",
            "world_frame": "camera",
            "cameras": cams,
            "quality": quality,
        }
    )


def _intrinsic_quality(name, obj, obs, K, dist, rvecs, tvecs, rms):
    """Measure fit and pose diversity; these are board-fit checks, not accuracy guarantees."""
    _rms(rms, f"{name} intrinsic RMS")
    normals, view_rms = [], {}
    for (fid, corners), rvec, tvec in zip(obs.items(), rvecs, tvecs, strict=True):
        projected = cv2.projectPoints(obj, rvec, tvec, K, dist)[0]
        error = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=2))))
        _rms(error, f"{name} board frame {fid} RMS", MAX_VIEW_RMS_PX)
        view_rms[str(fid)] = error
        normals.append(cv2.Rodrigues(rvec)[0][:, 2])
    span = max(
        np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), -1, 1)))
        for a, b in combinations(normals, 2)
    )
    if span < MIN_BOARD_NORMAL_SPAN_DEG:
        raise ValueError(
            f"Calibration quality gate: {name} board tilt spans only {span:.1f}°. Capture varied board tilts of at least {MIN_BOARD_NORMAL_SPAN_DEG:g}°."
        )
    return {
        "intrinsic_rms_px": float(rms),
        "per_view_rms_px": view_rms,
        "board_normal_span_deg": float(span),
    }


def calibrate(images, cols=9, rows=6, square=0.025):
    try:
        return _calibrate(images, cols, rows, square)
    except (cv2.error, np.linalg.LinAlgError) as exc:
        raise ValueError(
            "Calibration could not fit these board captures. Use sharp images with "
            "the full board visible, matching capture numbers, and varied board tilts."
        ) from exc


def _calibrate(images, cols=9, rows=6, square=0.025):
    if (
        not isinstance(images, dict)
        or not 2 <= len(images) <= 6
        or any(name not in CAMERA_NAMES for name in images)
    ):
        raise ValueError("Provide board images for 2–6 named cameras cam1 … cam6.")
    if (
        isinstance(cols, bool)
        or isinstance(rows, bool)
        or not isinstance(cols, int)
        or not isinstance(rows, int)
        or not 3 <= cols <= 20
        or not 3 <= rows <= 20
        or isinstance(square, bool)
        or not isinstance(square, (int, float, np.integer, np.floating))
        or not np.isfinite(square)
        or not 0.001 <= square <= 1
    ):
        raise ValueError("Check board dimensions and square size in meters.")
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.indices((cols, rows)).T.reshape(-1, 2) * square
    observations, cams = {}, {}
    for name, frames in images.items():
        if not isinstance(frames, dict) or not frames:
            raise ValueError(f"{name}: provide numbered board images.")
        obs, size = {}, None
        for fid, image in frames.items():
            if (
                not isinstance(image, np.ndarray)
                or image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[2] != 3
                or min(image.shape[:2]) < 2
            ):
                raise ValueError(f"{name}: invalid board image {fid}.")
            current = (image.shape[1], image.shape[0])
            if size is not None and size != current:
                raise ValueError(f"{name}: calibration image sizes differ.")
            size = current
            ok, corners = cv2.findChessboardCornersSB(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                (cols, rows),
                flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
            )
            if ok:
                obs[fid] = corners.astype(np.float32)
        if len(obs) < 12:
            raise ValueError(
                f"{name}: found board in {len(obs)} images; at least 12 are required."
            )
        # Repeated stills are not independent calibration poses, even if their fit is zero.
        variation = np.max(np.std(np.stack(list(obs.values())), axis=0))
        if variation < 1.0:
            raise ValueError(
                f"Calibration quality gate: {name} board captures repeat the same pose. Move and tilt the board between captures."
            )
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            [obj] * len(obs), list(obs.values()), size, None, None
        )
        quality = _intrinsic_quality(name, obj, obs, K, dist, rvecs, tvecs, rms)
        cams[name] = dict(
            K=K.tolist(),
            dist=dist.reshape(-1).tolist(),
            R=np.eye(3).tolist(),
            T=[0.0, 0.0, 0.0],
            image_size=list(size),
            **quality,
        )
        observations[name] = obs
    reference = sorted(cams)[0]
    rms_values, pairs = [], {}
    for name in sorted(cams):
        if name == reference:
            continue
        keys = sorted(set(observations[reference]) & set(observations[name]))
        if len(keys) < 12:
            raise ValueError(
                f"{reference}/{name}: only {len(keys)} shared board captures; need 12."
            )
        a, b = cams[reference], cams[name]
        if a["image_size"] != b["image_size"]:
            raise ValueError(
                "This calibration builder requires matching camera resolutions."
            )
        rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            [obj] * len(keys),
            [observations[reference][k] for k in keys],
            [observations[name][k] for k in keys],
            np.array(a["K"]),
            np.array(a["dist"]),
            np.array(b["K"]),
            np.array(b["dist"]),
            tuple(a["image_size"]),
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
        _rms(rms, f"{reference}/{name} stereo RMS")
        b.update(R=R.tolist(), T=T.reshape(-1).tolist())
        rms_values.append(float(rms))
        pairs[name] = len(keys)
    return validate(
        {
            "units": "meters",
            "world_frame": "camera",
            "cameras": cams,
            "board": {"cols": cols, "rows": rows, "square_m": square},
            "quality": {
                "accepted": True,
                "stereo_rms_px": max(rms_values),
                "paired_views": pairs,
                "source": "chessboard",
                "gates": {
                    "max_rms_px": MAX_RMS_PX,
                    "max_view_rms_px": MAX_VIEW_RMS_PX,
                    "min_board_normal_span_deg": MIN_BOARD_NORMAL_SPAN_DEG,
                },
            },
        }
    )
