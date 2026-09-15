import cv2
import numpy as np
from backend.geometry import project, solve_joint


def observations(point, cameras):
    return {
        n: {"xy": project(point, c)[0].tolist(), "confidence": 0.9}
        for n, c in cameras.items()
    }


def test_distorted_rotated_camera_roundtrip(calibration):
    cams = calibration["cameras"]
    cams["cam2"]["R"] = cv2.Rodrigues(np.array([0.02, -0.08, 0.01]))[0].tolist()
    point = np.array([0.2, 0.1, 4.0])
    result = solve_joint(observations(point, cams), cams)
    assert result["status"] == "observed"
    np.testing.assert_allclose(result["point"], point, atol=1e-6)


def test_bad_third_view_does_not_destroy_good_stereo(calibration):
    cams = calibration["cameras"]
    point = np.array([0.2, 0.1, 4.0])
    obs = observations(point, cams)
    obs["cam3"]["xy"][1] += 120
    result = solve_joint(obs, cams)
    np.testing.assert_allclose(result["point"], point, atol=1e-6)
    assert result["excluded"] == ["cam3"]
    assert set(result["cameras"]) == {"cam1", "cam2"}


def test_weak_geometry_rejected_even_with_zero_residual(calibration):
    cams = {k: v for k, v in calibration["cameras"].items() if k != "cam3"}
    cams["cam2"]["T"] = [-0.001, 0.0, 0.0]
    result = solve_joint(observations(np.array([0.0, 0.0, 4.0]), cams), cams)
    assert result["point"] is None


def test_negative_depth_is_rejected(calibration):
    cams = calibration["cameras"]
    assert (
        solve_joint(observations(np.array([0.0, 0.0, -4.0]), cams), cams)["point"]
        is None
    )


def test_only_edited_joint_observation_gains_authority(calibration):
    cams = {k: v for k, v in calibration["cameras"].items() if k != "cam3"}
    obs = observations(np.array([0.0, 0.0, 4.0]), cams)
    obs["cam2"]["confidence"] = 0.1
    assert solve_joint(obs, cams)["point"] is None
    obs["cam2"]["manual"] = True
    assert solve_joint(obs, cams)["status"] == "corrected"


def test_conflicting_manual_label_is_not_silently_ignored(calibration):
    cams = calibration["cameras"]
    point = np.array([0.2, 0.1, 4.0])
    obs = observations(point, cams)
    obs["cam3"]["xy"][1] += 120
    obs["cam3"]["manual"] = True
    assert solve_joint(obs, cams)["point"] is None


def test_epipolar_outlier_with_competing_stereo_pairs_is_rejected(calibration):
    cams = calibration["cameras"]
    for camera in cams.values():
        camera["dist"] = [0.0] * 5
    point = np.array([0.2, 0.1, 4.0])
    obs = observations(point, cams)
    obs["cam3"]["xy"][0] += 45
    result = solve_joint(obs, cams)
    assert result["point"] is None
    assert result["status"] == "rejected"
    assert "Ambiguous" in result["reason"]


def test_two_authoritative_views_resolve_epipolar_outlier(calibration):
    cams = calibration["cameras"]
    for camera in cams.values():
        camera["dist"] = [0.0] * 5
    point = np.array([0.2, 0.1, 4.0])
    obs = observations(point, cams)
    obs["cam3"]["xy"][0] += 45
    obs["cam1"]["manual"] = obs["cam2"]["manual"] = True
    result = solve_joint(obs, cams)
    assert result["status"] == "corrected"
    np.testing.assert_allclose(result["point"], point, atol=1e-6)
    assert result["excluded"] == ["cam3"]


def test_missing_label_only_excludes_its_view(calibration):
    cams = calibration["cameras"]
    point = np.array([0.2, 0.1, 4.0])
    obs = observations(point, cams)
    obs["cam3"] = {"xy": None, "confidence": 0.0, "manual": True}
    result = solve_joint(obs, cams)
    assert result["status"] == "observed"
    np.testing.assert_allclose(result["point"], point, atol=1e-6)
