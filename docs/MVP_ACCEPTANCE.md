# MVP release scope and acceptance

## Intended release

**Mocap Studio Web 0.2** is a local, single-operator application for reviewing one actor from 2–6 synchronized camera recordings. Its complete workflow is calibration → media import → 2D inference → actor review → multi-view 3D reconstruction → manual correction → durable project → export.

A browser is the user interface. Python inference and storage run on the same workstation. This release does not provide shared team accounts or a remotely hosted service.

## Release checks

| Operator outcome | Acceptance evidence |
| --- | --- |
| Install and start without editing source | Locked dependencies, model hashes, setup/start/doctor commands |
| Import actual recordings | Image and video codec tests, exact resolution checks, shared frame IDs and matching FPS |
| Keep source timing | Gapped image and strided video tests; timestamps survive project export/restore |
| Estimate real body landmarks | Published ONNX checkpoints run on a public image; model adapter returns COCO-17 observations |
| Reconstruct defensible 3D | Distortion-aware triangulation, positive-depth and angle checks, explicit missing data and excluded-view evidence |
| Repair wrong detections | Per-joint manual overlay, immediate reconstruction, undo, autosave, precise browser input |
| Avoid losing operator work | Atomic writes, revision checks, persistent inference progress and restart recovery, portable project backup |
| Manage local projects | Saved session reopening, rename, reversible archive and restore |
| Export a useful result | Timestamped metric skeleton JSON with coordinate convention, missing-data policy and per-joint quality |
| Understand uncertainty | Synthetic demo clearly identified; calibration status and 3D quality remain visible |

See [VALIDATION.md](VALIDATION.md) for the actual results and [ARCHITECTURE.md](ARCHITECTURE.md) for implementation details.

## Operating limits

- 300 sampled source frames, 500 MB per upload and 450 MB of stored camera images; use video stride or shorter clips for larger recordings.
- One process and one operator. Browser revision checks protect stale edits, but this is not a multi-user database or authentication system.
- Capture synchronization is supplied by the operator. The application cannot infer a hardware clock or correct unknown exposure offsets from filenames.
- Person identity is selected per view and frame. Automatic multi-person tracking and identity propagation are not included.
- COCO-17 body joints; derived body centers are labelled separately. No finger motion, heel/toe reconstruction, or character-rig retargeting.
- No synthetic motion completion, automatic bone constraints, smoothing, or foot locking.
- Model execution and deterministic geometry are verified. Real-rig accuracy is a separate field acceptance step.

## First real-rig acceptance

Before using a new camera rig for measurement, capture a held-out calibration target and one short synchronized person sequence. Compare reconstructed known distances against measured dimensions, inspect time alignment and left/right identity, and review representative occlusions. Record camera settings, calibration coverage, length error, missing-joint rate and operator correction time. The application cannot certify these properties from a low reprojection residual alone.

## Next release priorities

1. Measure error and correction effort on the actual target cameras and motions.
2. Add indexed frame storage and paged APIs for longer recordings.
3. Benchmark temporal/cross-view actor association using those captures.
4. Add measured synchronization offsets and, if required, a verified retargeting export.
5. Introduce authentication and per-user storage only if a shared deployment becomes a product requirement.
