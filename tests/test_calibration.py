"""Calibration schema boundaries and numerical board-fit regression checks."""

import io
import json

import cv2
import numpy as np
import pytest

from backend.calibration import (
    calibrate,
    read_calibration,
    validate,
    _intrinsic_quality,
)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cameras", []),
        ("cameras", None),
        ("cameras", [1, 2]),
        ("quality", None),
        ("quality", []),
        ("format", "mocap_calibration_v2"),
    ],
)
def test_malformed_top_level_calibration_is_actionable(calibration, field, value):
    calibration[field] = value
    with pytest.raises(ValueError):
        read_calibration(json.dumps(calibration), "rig.json")


@pytest.mark.parametrize(
    "field,value",
    [
        ("K", "invalid"),
        ("R", None),
        ("T", {"x": 1}),
        ("image_size", None),
        ("image_size", [True, 480]),
        ("image_size", [640.5, 480]),
        ("dist", {"x": 1}),
        ("intrinsic_rms_px", -1),
        ("intrinsic_rms_px", 2),
    ],
)
def test_malformed_camera_parameters_are_rejected(calibration, field, value):
    calibration["cameras"]["cam1"][field] = value
    with pytest.raises(ValueError):
        validate(calibration)


def test_camera_must_be_an_object(calibration):
    calibration["cameras"]["cam1"] = None
    with pytest.raises(ValueError, match="camera parameters"):
        validate(calibration)


@pytest.mark.parametrize(
    "value", [-0.1, float("nan"), float("inf"), None, "0.2", [0.2], True, 1.6]
)
def test_invalid_stereo_rms_is_rejected(calibration, value):
    calibration["quality"]["stereo_rms_px"] = value
    with pytest.raises(ValueError):
        validate(calibration)


@pytest.mark.parametrize("value", ["true", 1, [], {}])
def test_acceptance_flag_requires_boolean(calibration, value):
    calibration["quality"]["accepted"] = value
    with pytest.raises(ValueError, match="quality.accepted"):
        validate(calibration)


def test_nonzero_skew_is_rejected_instead_of_silently_ignored(calibration):
    calibration["cameras"]["cam1"]["K"][0][1] = 20
    with pytest.raises(ValueError, match="zero skew"):
        validate(calibration)


@pytest.mark.parametrize("names", [[], ["cam1", "cam1"], [1, 2], [["cam1", "cam2"]]])
def test_npz_camera_names_are_validated(names):
    stream = io.BytesIO()
    np.savez(stream, camera_names=np.array(names))
    with pytest.raises(ValueError, match="camera_names"):
        read_calibration(stream.getvalue(), "rig.npz")


def test_corrupt_npz_is_an_input_error():
    with pytest.raises(ValueError):
        read_calibration(b"PK\x03\x04corrupt", "rig.npz")


@pytest.mark.parametrize("square", [None, "0.025", True, float("nan")])
def test_invalid_board_scale_is_an_input_error(square):
    with pytest.raises(ValueError, match="square size"):
        calibrate({"cam1": {}, "cam2": {}}, square=square)


def test_degenerate_opencv_fit_is_an_actionable_input_error(monkeypatch):
    def fail(*args, **kwargs):
        raise cv2.error("synthetic failed board solve")

    monkeypatch.setattr(cv2, "findChessboardCornersSB", fail)
    images = {name: {0: np.zeros((480, 640, 3), np.uint8)} for name in ["cam1", "cam2"]}
    with pytest.raises(ValueError, match="could not fit"):
        calibrate(images)


def board_observations():
    obj = np.zeros((54, 3), np.float32)
    obj[:, :2] = np.indices((9, 6)).T.reshape(-1, 2) * 0.025
    K = np.array([[700.0, 0, 320], [0, 700.0, 240], [0, 0, 1.0]])
    points, images = {}, {}
    for camera, tx in [("cam1", 0.0), ("cam2", -0.7)]:
        points[camera], images[camera] = {}, {}
        for fid in range(18):
            rvec = np.array(
                [-0.35 + 0.14 * (fid % 6), -0.3 + 0.3 * (fid // 6), 0.08 * np.sin(fid)]
            )
            tvec = np.array(
                [
                    -0.10 + 0.06 * np.sin(fid),
                    -0.06 + 0.08 * np.cos(fid),
                    2.5 + 0.2 * np.sin(fid / 3),
                ]
            )
            tvec[0] += tx
            points[camera][fid] = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(5))[
                0
            ].astype(np.float32)
            images[camera][fid] = np.zeros((480, 640, 3), np.uint8)
    return points, images, K


def inject_board_detector(monkeypatch, points):
    corners = iter(p for camera in points.values() for p in camera.values())
    monkeypatch.setattr(
        cv2,
        "findChessboardCornersSB",
        lambda *args, **kwargs: (True, next(corners).copy()),
    )


def test_varied_board_calibration_recovers_metric_rig_and_records_evidence(monkeypatch):
    points, images, K = board_observations()
    inject_board_detector(monkeypatch, points)
    result = calibrate(images)
    assert result["quality"]["accepted"] is True
    assert result["quality"]["stereo_rms_px"] < 0.01
    np.testing.assert_allclose(result["cameras"]["cam2"]["T"], [-0.7, 0, 0], atol=0.001)
    np.testing.assert_allclose(result["cameras"]["cam1"]["K"], K, atol=0.1)
    for camera in result["cameras"].values():
        assert len(camera["per_view_rms_px"]) == 18
        assert camera["board_normal_span_deg"] > 20


def test_repeated_board_stills_fail_even_if_board_detected(monkeypatch):
    points, images, _ = board_observations()
    same = points["cam1"][0]
    monkeypatch.setattr(
        cv2, "findChessboardCornersSB", lambda *args, **kwargs: (True, same.copy())
    )
    with pytest.raises(ValueError, match="repeat the same pose"):
        calibrate(images)


def test_bad_stereo_fit_is_rejected_after_good_intrinsics(monkeypatch):
    points, images, K = board_observations()
    inject_board_detector(monkeypatch, points)
    monkeypatch.setattr(
        cv2,
        "stereoCalibrate",
        lambda *a, **k: (
            2.0,
            K,
            np.zeros(5),
            K,
            np.zeros(5),
            np.eye(3),
            np.array([[-0.7], [0], [0]]),
            None,
            None,
        ),
    )
    with pytest.raises(ValueError, match="stereo RMS exceeds"):
        calibrate(images)


def test_parallel_boards_fail_pose_diversity_check():
    obj = np.zeros((54, 3), np.float32)
    obj[:, :2] = np.indices((9, 6)).T.reshape(-1, 2) * 0.025
    K = np.array([[700.0, 0, 320], [0, 700.0, 240], [0, 0, 1.0]])
    rvecs = [np.array([0.0, 0.0, i * 0.03]) for i in range(12)]
    tvecs = [np.array([i * 0.01, 0.0, 2.0]) for i in range(12)]
    obs = {
        i: cv2.projectPoints(obj, r, t, K, np.zeros(5))[0]
        for i, (r, t) in enumerate(zip(rvecs, tvecs))
    }
    with pytest.raises(ValueError, match="board tilt"):
        _intrinsic_quality("cam1", obj, obs, K, np.zeros(5), rvecs, tvecs, 0.0)


def test_bad_individual_board_view_does_not_hide_inside_mean_rms():
    obj = np.zeros((54, 3), np.float32)
    obj[:, :2] = np.indices((9, 6)).T.reshape(-1, 2) * 0.025
    K = np.array([[700.0, 0, 320], [0, 700.0, 240], [0, 0, 1.0]])
    rvecs = [np.array([i * 0.03, 0.0, 0.0]) for i in range(12)]
    tvecs = [np.array([0.0, 0.0, 2.0]) for _ in range(12)]
    obs = {
        i: cv2.projectPoints(obj, r, t, K, np.zeros(5))[0]
        for i, (r, t) in enumerate(zip(rvecs, tvecs))
    }
    obs[5] += 4
    with pytest.raises(ValueError, match="frame 5 RMS"):
        _intrinsic_quality("cam1", obj, obs, K, np.zeros(5), rvecs, tvecs, 1.0)
