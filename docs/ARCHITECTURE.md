# Architecture and invariants

## Processing path

```text
Camera media + validated calibration
                  ↓
Timestamped source frames → RTMPose candidates → explicit actor selection
                  ↓
Raw observations + separate authoritative label overlay
                  ↓
Pair hypotheses → inlier cameras → weighted nonlinear refinement
                  ↓
Validated 3D joints + per-joint provenance/residuals
                  ↓
Camera reprojections / Three.js visualization / timestamped JSON export
```

### Core invariants

- Coordinates in `raw` always describe detector output for that inference run. Manual labels never modify that array.
- Each manual label is keyed by frame ID, camera ID, and COCO joint index. A `null` label explicitly excludes that observation.
- The `effective` layer overlays labels on predictions. Only individually edited observations gain manual authority.
- Every displayed 3D pose comes from that source frame. Editing runs the same geometry function used after inference.
- A joint needs at least two accepted views. Candidate scoring prioritizes inlier count, then residual and angle. Materially different, equally supported camera-pair hypotheses are rejected as ambiguous. Conflicting manual observations cannot simply become excluded outliers.
- Refinement minimizes weighted pixel reprojection residuals against accepted cameras using a robust loss. It is followed by independent residual/depth/angle checks.
- Timestamps are source time. Skipped images or stride never compress time. There is no retiming or implicit interpolation.
- All JSON output uses strict finite values and `null` for missing data.
- Session writes use a same-directory temporary file plus atomic replacement. Writes flush data before atomic replacement. In-process locks serialize mutations; revision checks reject stale browser edits.
- A single background worker serializes inference. Cancellation is checked between views; completed frames remain saved. Pending partial-frame work is not committed.
- Jobs are saved under the session storage root in `.jobs/`. Each completed frame and its job marker share an atomic checkpoint. Startup reconciles markers and exposes unfinished work as interrupted; resume skips completed frames. Shutdown requests cancellation and waits for the current model call.
- Saved project ZIPs include calibration, images, model predictions, person candidates and selection, labels, timestamps, and metrics. Restore validates paths, archive size, coordinates, calibration, and image sizes, creates a new session identity, recomputes geometry, and clears prior undo history.

## Important interfaces

`backend/geometry.py` contains the pure geometry operations. `backend/calibration.py` defines the validated camera representation and imports old NPZ formats at the boundary. `backend/detector.py` adapts model output into 17 pixel-coordinate observations. `backend/store.py` owns on-disk sessions. `backend/app.py` exposes them to the browser.

The frontend's `types.ts` describes session, frame, observation and quality payloads. `CameraView.tsx` implements SVG coordinates, dragging and zoom. `Skeleton3D.tsx` applies the explicit world orientation to Three.js. `App.tsx` coordinates sessions, jobs, selection, playback and editor controls. `Dialogs.tsx` handles capture/calibration imports.

## Conventions

Camera transform: `X_camera = R @ X_world + T`. Translation and 3D points are meters; 2D positions and residuals are pixels. Image size is `[width, height]`. Distortion follows OpenCV's standard pinhole model.

`world_frame: camera` means x-right, y-down, z-forward in the reference camera. Viewer conversion is `(x, -y, -z)` to Three.js Y-up. `world_frame: z_up` maps `(x, z, -y)` to the viewer. Export retains original coordinates and the explicit world-frame declaration.

## Deliberate limitations and next steps

- One local process; file storage is not a multi-process database. Do not run multiple server workers against the same data directory.
- Sessions are loaded as a whole. Large recordings need indexed storage, pagination and resumable media ingest.
- Videos assume constant rate and synchronized starts; hardware timestamps and synchronization estimation are future work.
- Identity selection is per view/frame, not a temporal multi-person tracking system.
- No model fine-tuning, confidence boosting from prior edits, smoothing, kinematic constraints, or learned motion completion.
- No automatic Blender rigging or character retargeting. Use the explicit exported schema for a separate verified integration.
- Imported calibration without quality evidence is marked as imported; transform validation cannot prove physical accuracy. Recalibrate or verify the rig with known-distance/held-out targets before trusting capture results.
