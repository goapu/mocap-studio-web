# MVP validation

Validated locally on **16 September 2026**, macOS arm64, Python 3.13.5, Node.js 24.14.1.

## Automated checks

| Check | Result |
| --- | --- |
| Backend regression suite | **80 passed**, 1 optional real-model test skipped in the default run |
| Real-model API integration | **1 passed** separately with local ONNX checkpoints and a public RTMLib image |
| Frontend workflow tests | **8 passed** |
| Python lint and formatting | Passed |
| Frontend formatting, TypeScript, production build | Passed |
| Installed Python dependency compatibility | Passed (`pip check`) |
| Runtime imports, model hashes, build and writable storage | Passed (`scripts/manage.py doctor`) |
| Frontend dependency audit | 0 vulnerabilities reported during the tested installation |

Backend coverage includes calibration schema and fit gates, degenerate/ambiguous geometry, distortion, positive depth, manual authority, gapped timing, actual image/video codecs, stale revisions, atomic frame checkpoints, restart recovery, cancellation, resume, session archiving, backup round trips, path validation and storage limits.

Frontend tests cover interrupted-job discovery/resume, initial connection retry, remembered frame, archive restoration, keyboard editing, disabled editing during processing, numeric drafts across frame changes and drag cancellation on frame changes.

The real-model integration imports two views, runs the actual detector/pose model through the API, selects a person, and restores the project including candidate detections. It deliberately uses the same image for both virtual cameras and verifies missing 3D output from absent stereo parallax. **It does not claim those images are a real stereo capture.**

## Real 2D inference smoke test

The public RTMLib `demo.jpg` (950 × 641) produced nine person candidates, each with 17 joint observations. A local CPU run measured approximately **0.78 seconds** for model initialization plus first inference and **0.45 seconds** for a warm call. These single-image observations are not a throughput benchmark or an accuracy evaluation.

The image lives only in the separate testing workspace; no public photograph or user recording is bundled in Git. The model URLs and verified SHA-256 hashes are recorded in `models.lock.json`.

To rerun with your own local test image:

```bash
.venv/bin/python scripts/smoke_inference.py /path/to/person.jpg
MOCAP_TEST_IMAGE=/path/to/person.jpg .venv/bin/python -m pytest tests/test_inference_integration.py
```

The optional API integration assumes a person is visible in the supplied image.

## Browser acceptance

The final browser build was exercised against the live local backend. Verification includes the misaligned demo wrist, camera zoom, precise correction, immediate 3D feedback, persistence after reload and session management. The demo correction changes accepted views from cam1/cam3 to all three cameras and removes the frame's review flag.

## Practical limits

- Real synchronized human footage from the intended rig has not been supplied; metric accuracy, sync tolerance and operator correction effort still require field acceptance.
- The calibration gates were tested with numerical fixtures, malformed inputs and controlled board-observation scenarios. Real camera hardware and board capture are not certified by those tests.
- Windows/Linux installation and remote CI have not been executed. CI configuration is included for the first repository push.
- The build reports a large Three.js bundle warning. This is served locally and the build succeeds; bundle splitting is a future optimization.
- The backend tests emit two upstream Starlette/httpx/AnyIO deprecation warnings. They do not affect test results.
