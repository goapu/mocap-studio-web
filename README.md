
# Mocap Studio Web

**A local workstation for reviewing synchronized camera footage and producing traceable 3D motion.**

Mocap Studio combines multi-camera pose analysis, calibrated 3D reconstruction, and operator correction in one browser workspace. Your footage, model inference, annotations, and saved sessions remain on the workstation; no cloud account or desktop application is required.

## What you can do

- Import 2–6 synchronized videos or numbered image sequences.
- Calibrate a camera rig from chessboard captures or import an existing JSON/NPZ calibration.
- Run RTMPose-M body landmark analysis and explicitly match people across camera views.
- Review the reconstructed 3D skeleton, correct joints directly in each camera view, and see the result update immediately.
- Preserve corrections through re-analysis, recover interrupted work, and resume from unfinished frames.
- Save a portable project backup or export timestamped, metric 3D motion JSON with quality evidence.

The bundled reference sequence is a safe place to learn the correction workflow before importing a real capture.

## Run it in three commands

From the repository root:

Use **Python 3.12 or 3.13**, Node.js 22+ and npm. The validated development machine used Python 3.13.5 and Node.js 24.14.1 on macOS arm64.

bash
python3 scripts/manage.py setup
python3 scripts/manage.py doctor
python3 scripts/manage.py start
Open http://127.0.0.1:8765. On Windows, use python if that is your Python command. setup installs the locked Python environment, builds the browser UI, and downloads verified weights. start binds only to the local workstation. Use start --port 8766 if another program already uses port 8765.

For an editor-only installation with the synthetic demo, run setup --skip-models. Install the models later with .venv/bin/python scripts/download_models.py (Windows: .venv\Scripts\python.exe). Internet access is needed during setup; operation is local afterward.

Bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python scripts/download_models.py
npm --prefix frontend ci
npm --prefix frontend run build
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
On Windows, activate with .venv\Scripts\Activate.ps1 instead.

Model downloads are needed once and cached under .local/models/ by default. The downloader verifies the ONNX weights against the recorded SHA-256 hashes. The combined downloads are approximately 140 MB compressed. You can run the synthetic demo without downloading models.

The UI and the API share the same local server. Keep the terminal running while using the application. There is no automatic startup service.

Development
Use two terminals with the virtual environment active:

Bash
# Terminal 1, repository root
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8765

# Terminal 2, repository root
npm --prefix frontend run dev
Open Vite's displayed local URL. The dev server proxies /api to port 8765. React fast refresh updates the UI while editing.

Bash
python3 scripts/manage.py test

# Optional contributor lint/format checks
python -m pip install -r requirements-dev.txt
ruff check backend scripts tests
ruff format --check backend scripts tests
Your first capture
Calibrate the rig. In Camera calibration, provide at least 12 synchronized chessboard images for every camera. Use the measured square size in meters. The application accepts a calibration only after intrinsic fit, stereo fit, and board-pose diversity checks pass.

Import synchronized media. Select one video or a numbered image sequence for each camera and attach the corresponding calibration. Every video must have matching constant FPS and frame-zero alignment; every image must use the calibration resolution.

Run pose analysis. The job runs locally with progress and cancellation. If several people appear, choose the same person in each camera panel before reviewing 3D motion.

Review and refine. Use the review queue to visit flagged frames. Drag a joint, use the precise pixel controls, or place a missing observation. Orange crosses show the current 3D reprojection. Saved labels are never silently overwritten by a later model run.

Deliver the result. Download a project backup when you need to continue work later. Export 3D motion JSON when you need timestamped joint positions and quality evidence for another tool.

For a guided dry run, open the included synthetic demo and use frame 20 to correct the deliberately offset right wrist in cam2.

Capture requirements
Videos must be synchronized, constant-FPS, and aligned at frame zero. This release does not measure or repair synchronization. Variable-frame-rate videos should be converted to constant FPS first.

All video cameras in a session must report matching FPS. Camera image resolution must match that camera's calibration exactly.

For images, provide the true source FPS and shared numbered filenames such as frame001.jpg, frame004.jpg. Gaps remain gaps in time. Files can be selected separately for each camera, so camera labels do not have to appear in filenames.

MVP limits: 300 sampled frames, 500 MB per upload, and 450 MB of extracted camera images per session. Video stride reduces sampling density while preserving original frame timestamps.

The chessboard builder currently requires equal image dimensions across cameras. Imported calibrated cameras may have different resolutions.

Keep one intended actor in the capture volume when possible. Multiple-person selections are per frame and require operator review.

Data and accuracy boundaries
This MVP delivers the complete local operator workflow. It is not yet a metrologically validated capture system. Real RTMPose 2D inference was verified on a public image; real synchronized multi-camera human capture accuracy has not yet been benchmarked.

3D positions use calibrated multi-view geometry, not an inferred single-camera body shape. A joint needs at least two usable views. Missing joints stay null; there is no automatic gap filling, motion smoothing, bone-length enforcement, or foot locking. This preserves evidence and avoids introducing the timing/contact defects found in the original scripts.

Reprojection error measures consistency with image observations. It does not certify depth accuracy, actor identity, synchronization, anatomical correctness, or calibration coverage. Inspect accepted/excluded views as well as the rendered skeleton.

The output is COCO-17 body pose plus derived pelvis, neck, spine, and head positions. Detailed fingers, heels/toes, and production-character retargeting are outside this release. A head-to-neck segment is displayed using the shoulder midpoint; it is a visual derived segment, not a separately detected landmark.

The 3D grid is a reference aid. With world_frame: camera, its height follows the observed skeleton and it is not a measured floor. A z_up calibration uses the supplied world Z=0 plane.

Repository layout
Plaintext
backend/                  FastAPI routes, calibration, geometry, model adapter, storage
frontend/src/             Typed React editor, camera interaction, Three.js viewer
tests/                    Geometry and workflow regression tests
scripts/                  Setup/start/doctor, model downloader, inference smoke check
docs/                     Architecture, calibration format, validation evidence
examples/                 Calibration schema example (not a real rig calibration)
.github/workflows/ci.yml   Backend/frontend tests, lint, formatting and build
requirements.txt          Direct dependency versions
requirements.lock.txt     Tested complete Python dependency set
requirements-dev.txt      Contributor lint dependency
frontend/package-lock.json
.local/                   Ignored recordings, saved sessions, model weights
The original desktop source is preserved outside this new repository. This implementation uses one reconstruction backend for import, inference, edits, quality reporting, and export.

The public repository is available at goapu/mocap-studio-web. Recordings, model binaries, environments, caches, and generated builds are ignored.

Release evidence and operation
MVP scope and acceptance criteria

Validation results

Architecture and data invariants

Calibration schema and capture guidance

Operator troubleshooting

Configuration
MOCAP_DATA_DIR and MOCAP_MODEL_DIR optionally relocate session and model storage. MOCAP_PYTHON optionally selects an existing environment for the launcher. Set overrides as environment variables; .env.example documents them but is not loaded automatically.

The server is intended for a single local workstation and binds to loopback. It rejects browser requests with nonlocal origins. It has no authentication, multi-user isolation, remote deployment configuration, or public hosting setup.

Model choice and provenance
RTMPose-M is a practical CPU baseline with an explicit ONNX path and established body-pose implementation. It is not claimed to be universally the most accurate open model. YOLOX-M proposes person boxes; RTMPose-M estimates the joints. The detector is isolated behind an adapter for future benchmarked replacements.

See RTMPose, RTMLib, and third-party notices. Code and checkpoint provenance are distinct; retain upstream notices and review checkpoint terms for your intended distribution.

License
This project is licensed under the MIT License.

Note that the model weights (RTMPose-M, YOLOX-M) and certain third-party dependencies may be subject to their own respective licenses. Please review THIRD_PARTY_NOTICES.md for details on upstream code and checkpoint provenance before distributing derivative works.
