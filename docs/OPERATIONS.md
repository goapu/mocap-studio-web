# Local operation and recovery

## Start and stop

From the repository root, run `python3 scripts/manage.py start`. Leave that terminal running and open the displayed local URL. Stop with Ctrl+C. Use one server process; multiple workers do not share the file locks and job queue.

Run `python3 scripts/manage.py doctor` to check Python imports, package compatibility, model hashes, the browser build and writable storage. The installed environment and local data are excluded from Git.

## Storage and backups

Defaults:

- `.local/sessions/`: imported frames encoded as JPEG, calibration, observations, manual labels, saved session state, and durable job metadata. Keep original source recordings separately if you need their full image fidelity.
- `.local/models/`: downloaded ONNX checkpoints and their manifest.
- `.venv/`: installed Python environment.
- `frontend/node_modules/`, `frontend/dist/`: installed dependencies and browser build.

Use **Save project ZIP** for a portable project backup. It contains camera images and labels, so store it with your recording data. The 3D motion JSON is an output, not a complete editable project backup. A restored ZIP creates a new session and recomputes geometry; undo history starts empty.

Archiving a session only removes it from the active list. It remains recoverable through Archived sessions. To back up all local state, stop the server and copy the complete session storage directory. A Git push does not back up recordings, weights, sessions, or environments.

## Interrupted inference

Job progress is durable. On restart, queued/running jobs become interrupted; reopening the associated session offers Resume. Resume uses the unfinished frame list. Saved, completed frames and their manual labels remain intact. Cancelling a running job finishes the current model call, discards the incomplete frame, and keeps prior completed frames.

If model files are missing or damaged, rerun the model downloader after removing only the file named by the checksum error. `doctor` verifies the hashes against `models.lock.json`.

## Common problems

| Symptom | Action |
| --- | --- |
| Browser cannot connect | Keep the start terminal running; inspect its error. Try another local port if occupied. |
| Missing browser build or packages | Run `python3 scripts/manage.py setup` from this repository. |
| Model unavailable | Run the model downloader using the installed virtual environment, then reopen the page. |
| Calibration import rejected | Check JSON structure, camera image dimensions, world convention and quality. Use the builder for a new rig. |
| No matching frames | Use the same numbered source frame IDs in all camera sequences. |
| Video FPS mismatch | Supply synchronized constant-rate recordings with matching FPS and frame-zero alignment. |
| Resolution mismatch | Use the same uncropped camera resolution used for calibration; resizing only the footage invalidates pixel geometry. |
| Person selector is shown | Select the same actor in each view. Candidates are ranked independently per view. |
| 3D joint remains missing | Supply two consistent observations, sufficient baseline, valid calibration and correct identity. Inspect excluded cameras and low-confidence observations. |
| Manual correction creates a missing joint | Conflicting labels remain saved. Correct the disagreement instead of lowering thresholds to force a solution. |
| Session changed in another tab | Reload the session before editing; stale edits are rejected. |
| Import exceeds limits | Shorten the clip or increase video stride. Source timestamps are retained. |

## Privacy and deployment

After setup, browser assets, inference and session data operate locally. There are no third-party font requests or cloud inference calls. This release has no account system and is intended for loopback use. A shared network service requires a separate authentication, storage-isolation and deployment design.
