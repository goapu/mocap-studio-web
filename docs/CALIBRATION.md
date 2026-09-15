# Calibration format

The canonical JSON is `mocap_calibration_v1`, illustrated by `examples/calibration.example.json`. The example is a synthetic geometry/schema reference and must not be used for real footage.

Required top-level fields:

| Field | Meaning |
|---|---|
| `units` | Exactly `meters` |
| `world_frame` | `camera` or `z_up` |
| `cameras` | Named `cam1` through `cam6`; at least two |
| `quality` | Calibration source and acceptance evidence where available |

Each camera requires a 3×3 intrinsic matrix `K`, distortion vector `dist`, proper 3×3 rotation `R`, three-element translation `T`, and integer `[width,height]` `image_size`.

The parser checks finite parameters, positive focal lengths, zero skew (required by the OpenCV pinhole model), valid rotation matrices, distortion dimensions, image sizes, and a nonzero rig baseline. Actual frames must match each camera's calibration resolution.

## Legacy NPZ

- Named-camera format: `camera_names`, and `{camera}_K`, `{camera}_dist`, `{camera}_R`, `{camera}_T`, `{camera}_image_size`.
- Legacy stereo format: `K1`, `dist1`, `K2`, `dist2`, `R`, `T`, and `image_size`.
- If both formats are present, their relative transforms must agree. The stale mixed geometry from the old marker-refresh script is rejected.
- An NPZ with nonidentity first-camera extrinsics is rejected because its world orientation is ambiguous. Convert it to canonical JSON with explicit orientation.
- Missing image dimensions are rejected; they cannot safely be inferred from arbitrary new footage.
- Stored intrinsic or stereo RMS above 1.5 px, invalid/negative RMS, or explicit failed acceptance is rejected.

## Built-in chessboard builder

Use matching image numbers across cameras, 12 or more jointly visible captures per camera pair, measured square dimensions, and varied board orientations. Intrinsics are estimated per camera; extrinsics are estimated from common observations relative to the first camera. Intrinsic and stereo RMS must each be at most 1.5 px. The worst individual board view must fit within 3 px RMS. Repeated still images are rejected, and estimated board normals must span at least 8° to avoid a degenerate calibration from one pose. Vary board position, distance, and tilt while keeping it clearly visible in every camera.

This is a board-fit gate, not proof of human pose accuracy. There is no automatic floor detection, ChArUco refresh, rolling-shutter compensation, or synchronization estimation in this release.
