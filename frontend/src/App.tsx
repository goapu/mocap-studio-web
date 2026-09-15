import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  Box,
  Camera,
  Check,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  Download,
  FolderOpen,
  Grid2X2,
  Layers,
  LoaderCircle,
  Pause,
  Play,
  Plus,
  ScanLine,
  Settings2,
  SkipBack,
  Undo2,
  Upload,
  X,
  MousePointer2,
  Crosshair,
  AlertCircle,
  SlidersHorizontal,
} from "lucide-react";
import type { Session, Summary, Job, Calibration } from "./types";
import { api, json, jointLabel } from "./types";
import CameraView from "./CameraView";
import Skeleton3D from "./Skeleton3D";
import { ImportDialog, CalibrationDialog, Modal } from "./Dialogs";

function downloadJSON(data: unknown, name: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function App() {
  const [session, setSession] = useState<Session | null>(null),
    [saved, setSaved] = useState<Summary[]>([]),
    [index, setIndex] = useState(0),
    [selected, setSelected] = useState(10);
  const [cameraA, setCameraA] = useState("cam1"),
    [cameraB, setCameraB] = useState("cam2");
  const [importOpen, setImportOpen] = useState(false),
    [calibrateOpen, setCalibrateOpen] = useState(false),
    [sessionsOpen, setSessionsOpen] = useState(false),
    [settingsOpen, setSettingsOpen] = useState(false);
  const [calibration, setCalibration] = useState<File | null>(null),
    [modelReady, setModelReady] = useState(false),
    [busy, setBusy] = useState(false),
    [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [playing, setPlaying] = useState(false),
    [raw, setRaw] = useState(false),
    [projection, setProjection] = useState(true),
    [place, setPlace] = useState(false);
  const [archivedList, setArchivedList] = useState(false);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [sessionName, setSessionName] = useState("");
  const [sessionError, setSessionError] = useState("");
  const [settingsError, setSettingsError] = useState("");
  const [starting, setStarting] = useState(true);
  const [recovering, setRecovering] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [pollRetry, setPollRetry] = useState(0);
  const [connectionLost, setConnectionLost] = useState(false);
  const loading = useRef(false);
  const mutation = useRef(false);
  const savedRequest = useRef(0);
  const epoch = useRef(0);
  const sessionRef = useRef(session);
  sessionRef.current = session;
  const activeJob =
    job && ["queued", "running"].includes(job.status) ? job : null;
  const disabled = starting || busy || recovering || !!activeJob;
  const navigationLocked = dragging || busy;
  const frame = session?.frames[Math.min(index, session.frames.length - 1)];
  async function refreshSaved(archived = archivedList) {
    const request = ++savedRequest.current;
    try {
      const list = await api<Summary[]>(
        `/sessions${archived ? "?archived=true" : ""}`,
      );
      if (request === savedRequest.current) setSaved(list);
    } catch (e) {
      if (request === savedRequest.current)
        setSessionError((e as Error).message);
    }
  }

  function accept(updated: Session, token: number) {
    if (token !== epoch.current || updated.id !== sessionRef.current?.id)
      return;
    setSession((previous) =>
      previous && previous.revision > updated.revision ? previous : updated,
    );
  }
  function lock() {
    if (mutation.current) return false;
    mutation.current = true;
    setBusy(true);
    setPlaying(false);
    setError("");
    return true;
  }
  function unlock() {
    mutation.current = false;
    setBusy(false);
  }
  async function discoverJob(sid: string, token: number) {
    setRecovering(true);
    try {
      const latest = await api<Job | null>(`/sessions/${sid}/job`);
      if (epoch.current === token) setJob(latest);
    } catch (e) {
      if (epoch.current === token)
        setError(
          `Could not restore processing status. ${(e as Error).message}`,
        );
    } finally {
      if (epoch.current === token) setRecovering(false);
    }
  }
  function load(s: Session, frameId?: number) {
    const token = ++epoch.current;
    sessionRef.current = s;
    setSession(s);
    setIndex(
      Math.max(
        0,
        s.frames.findIndex((f) => f.id === frameId),
      ),
    );
    setPlaying(false);
    setDragging(false);
    setJob(null);
    setConnectionLost(false);
    const names = Object.keys(s.calibration.cameras);
    setCameraA(names[0]);
    setCameraB(names[1]);
    setError("");
    setStarting(false);
    refreshSaved();
    void discoverJob(s.id, token);
  }
  async function initialize() {
    if (loading.current) return;
    loading.current = true;
    setStarting(true);
    setError("");
    try {
      const [health, list] = await Promise.all([
        api<{ model_ready: boolean }>("/health"),
        api<Summary[]>("/sessions"),
      ]);
      setModelReady(health.model_ready);
      setSaved(list);
      let remembered: { id?: string; frameId?: number } = {};
      try {
        remembered = JSON.parse(
          localStorage.getItem("mocap.workspace") || "{}",
        );
      } catch {
        /* Storage may be unavailable. */
      }
      const sid = list.find((s) => s.id === remembered.id)?.id || list[0]?.id;
      const s = sid
        ? await api<Session>(`/sessions/${sid}`)
        : await api<Session>("/demo", json("POST"));
      load(s, sid === remembered.id ? remembered.frameId : undefined);
    } catch (e) {
      setError((e as Error).message);
      setStarting(false);
    } finally {
      loading.current = false;
    }
  }
  useEffect(() => {
    void initialize();
  }, []);
  useEffect(() => {
    if (!session || !frame) return;
    try {
      localStorage.setItem(
        "mocap.workspace",
        JSON.stringify({ id: session.id, frameId: frame.id }),
      );
    } catch {
      /* The server still saves the project. */
    }
  }, [session?.id, frame?.id]);
  useEffect(() => {
    if (!notice) return;
    const id = setTimeout(() => setNotice(""), 4500);
    return () => clearTimeout(id);
  }, [notice]);
  useEffect(() => {
    if (!activeJob) return;
    const token = epoch.current;
    let cancelled = false;
    const timer = setTimeout(
      async () => {
        try {
          const next = await api<Job>(`/jobs/${activeJob.id}`);
          if (cancelled || token !== epoch.current) return;
          setConnectionLost(false);
          if (!["queued", "running"].includes(next.status)) {
            const updated = await api<Session>(`/sessions/${next.session_id}`);
            if (cancelled || token !== epoch.current) return;
            accept(updated, token);
            if (next.status === "failed") setError(next.message);
            else setNotice(next.message);
          }
          setJob(next);
        } catch {
          if (cancelled || token !== epoch.current) return;
          setConnectionLost(true);
          setPollRetry((n) => n + 1);
        }
      },
      connectionLost ? 2500 : 700,
    );
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [job, pollRetry, connectionLost]);
  useEffect(() => {
    if (!playing || !session || navigationLocked || disabled) return;
    if (index >= session.frames.length - 1) {
      setPlaying(false);
      return;
    }
    const delay = Math.max(
      10,
      (session.frames[index + 1].timestamp - session.frames[index].timestamp) *
        1000,
    );
    const timer = setTimeout(() => setIndex((i) => i + 1), delay);
    return () => clearTimeout(timer);
  }, [playing, index, session, navigationLocked, disabled]);
  const undo = useCallback(async () => {
    if (!session || disabled || !lock()) return;
    const token = epoch.current;
    try {
      accept(
        await api<Session>(
          `/sessions/${session.id}/undo?revision=${session.revision}`,
          json("POST"),
        ),
        token,
      );
      setNotice("Last joint correction undone.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      unlock();
    }
  }, [session, disabled]);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (
        (e.target as Element).closest(
          "input,select,textarea,dialog,button,[role='button']",
        ) ||
        !session ||
        navigationLocked
      )
        return;
      if (e.key === "ArrowRight") {
        e.preventDefault();
        setIndex((i) => Math.min(session.frames.length - 1, i + 1));
        setPlaying(false);
      }
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        setIndex((i) => Math.max(0, i - 1));
        setPlaying(false);
      }
      if (e.code === "Space" && !disabled) {
        e.preventDefault();
        setPlaying((p) => !p);
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "z") {
        e.preventDefault();
        void undo();
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [session, undo, navigationLocked, disabled]);
  async function edit(
    frameId: number,
    camera: string,
    joint: number,
    value: [number, number] | null,
    reset = false,
  ) {
    if (!session || !frame || disabled || frame.id !== frameId || !lock())
      return;
    const token = epoch.current;
    try {
      accept(
        await api<Session>(
          `/sessions/${session.id}/joint`,
          json("PATCH", {
            frame_id: frameId,
            camera,
            joint,
            value,
            reset,
            revision: session.revision,
          }),
        ),
        token,
      );
      setNotice(
        reset
          ? "Model prediction restored."
          : "Correction saved. 3D skeleton updated.",
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      unlock();
    }
  }
  async function actor(camera: string, actorIndex: number) {
    if (!session || !frame || disabled) return;
    if (
      Object.keys(frame.views[camera].edits).length &&
      !confirm(
        "Selecting a different person clears manual labels for this camera frame. Continue?",
      )
    )
      return;
    if (!lock()) return;
    const token = epoch.current;
    try {
      accept(
        await api<Session>(
          `/sessions/${session.id}/actor`,
          json("POST", {
            camera,
            index: actorIndex,
            frame_id: frame.id,
            revision: session.revision,
          }),
        ),
        token,
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      unlock();
    }
  }
  async function detect(current = false) {
    if (!session || disabled || !lock()) return;
    const token = epoch.current;
    try {
      const next = await api<Job>(
        `/sessions/${session.id}/detect?revision=${session.revision}${current ? "&frame_id=" + frame?.id : ""}`,
        json("POST"),
      );
      if (token === epoch.current) setJob(next);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      unlock();
    }
  }
  async function resume() {
    if (!session || !job?.can_resume || disabled || !lock()) return;
    const token = epoch.current;
    try {
      const next = await api<Job>(
        `/jobs/${job.id}/resume?revision=${session.revision}`,
        json("POST"),
      );
      if (token === epoch.current) setJob(next);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      unlock();
    }
  }
  async function reloadSession() {
    if (!session || !lock()) return;
    try {
      const [updated, health] = await Promise.all([
        api<Session>(`/sessions/${session.id}`),
        api<{ model_ready: boolean }>("/health"),
      ]);
      setModelReady(health.model_ready);
      load(updated, frame?.id);
      setNotice("Latest saved session loaded.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      unlock();
    }
  }
  async function freshDemo() {
    if (!lock()) return;
    try {
      load(await api<Session>("/demo", json("POST")), 20);
      setNotice("Demo ready. Try correcting the right wrist in cam2.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      unlock();
    }
  }
  function nextReview() {
    if (!session || navigationLocked) return;
    for (let off = 1; off <= session.frames.length; off++) {
      const i = (index + off) % session.frames.length;
      if (session.frames[i].quality.needs_review) {
        setIndex(i);
        setPlaying(false);
        return;
      }
    }
    setNotice("No frames are flagged by the current geometry checks.");
  }
  function built(cal: Calibration) {
    setCalibration(
      new File([JSON.stringify(cal)], "camera_calibration.json", {
        type: "application/json",
      }),
    );
    downloadJSON(cal, "camera_calibration.json");
    setCalibrateOpen(false);
    setImportOpen(true);
    setNotice("Calibration accepted and attached to your new capture.");
  }
  async function restore(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file || !lock()) return;
    const data = new FormData();
    data.append("project", file);
    try {
      load(await api<Session>("/restore", { method: "POST", body: data }));
      setSessionsOpen(false);
      setNotice("Project restored, including images and labels.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      unlock();
    }
  }
  async function manageSession(
    summary: Summary,
    action: "rename" | "archive" | "unarchive",
  ) {
    if (!lock()) return;
    setSessionError("");
    try {
      const current = await api<Session>(`/sessions/${summary.id}`);
      const updated =
        action === "rename"
          ? await api<Session>(
              `/sessions/${summary.id}`,
              json("PATCH", { name: sessionName, revision: current.revision }),
            )
          : await api<Session>(
              `/sessions/${summary.id}/${action}?revision=${current.revision}`,
              json("POST"),
            );
      if (sessionRef.current?.id === updated.id) accept(updated, epoch.current);
      setRenaming(null);
      await refreshSaved();
      setNotice(
        action === "rename"
          ? "Session renamed."
          : action === "archive"
            ? "Session archived. Restore it from the archived sessions list."
            : "Session restored to your workspace.",
      );
    } catch (e) {
      setSessionError((e as Error).message);
    } finally {
      unlock();
    }
  }
  const reviewCount =
    session?.frames.filter((f) => f.quality.needs_review).length || 0;
  const editedCount =
    session?.frames.reduce(
      (a, f) =>
        a +
        Object.values(f.views).reduce(
          (n, v) => n + Object.keys(v.edits).length,
          0,
        ),
      0,
    ) || 0;
  const names = session ? Object.keys(session.calibration.cameras) : [];
  const joint = frame?.pose[selected];
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="Mocap Studio home">
          <div className="brand-mark">
            <Activity size={24} />
          </div>
          <span>
            MOCAP<span className="brand-sub">STUDIO / WEB</span>
          </span>
        </a>
        <div className="nav-section-label">WORKSPACE</div>
        <button className="nav-item active">
          <ScanLine size={19} />
          Pose editor<span className="nav-count">01</span>
        </button>
        <button
          className="nav-item"
          disabled={disabled}
          onClick={() => {
            setSessionError("");
            refreshSaved();
            setSessionsOpen(true);
          }}
        >
          <FolderOpen size={19} />
          Saved sessions
        </button>
        <button
          className="nav-item"
          onClick={() => setCalibrateOpen(true)}
          disabled={disabled}
        >
          <Camera size={19} />
          Camera calibration
        </button>
        <div className="sidebar-line" />
        <div className="nav-section-label">CURRENT SESSION</div>
        <div className="session-side">
          <span className="small-folder">
            <Layers size={19} />
          </span>
          <strong>{session?.name || "Preparing workspace…"}</strong>
          <small>
            {session?.source === "synthetic"
              ? "Synthetic reference sequence"
              : "Synchronized capture"}
          </small>
        </div>
        <dl className="session-facts">
          <div>
            <dt>Cameras</dt>
            <dd>{names.length || "—"}</dd>
          </div>
          <div>
            <dt>Frames</dt>
            <dd>{session?.frames.length || "—"}</dd>
          </div>
          <div>
            <dt>Source rate</dt>
            <dd>{session ? session.fps.toFixed(2) + " fps" : "—"}</dd>
          </div>
          <div>
            <dt>Manual labels</dt>
            <dd>{editedCount}</dd>
          </div>
        </dl>
        <button
          className="import-side"
          onClick={() => setImportOpen(true)}
          disabled={disabled}
        >
          <Plus size={17} />
          Import capture
        </button>
        <div className="sidebar-bottom">
          <div className="model-badge">
            <Box size={19} />
            <span>
              RTMPose-M
              <small>
                {modelReady
                  ? "Model available · CPU"
                  : "Model download required"}
              </small>
            </span>
          </div>
          {!modelReady && (
            <button
              className="text-button"
              disabled={busy}
              onClick={reloadSession}
            >
              Refresh model status
            </button>
          )}
          <p>Local processing. Camera footage stays on this workstation.</p>
          <span className="version">LOCAL EDITION 0.2</span>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <ChevronRight size={14} />
            <span>Pose editor</span>
          </div>
          <div className="top-actions">
            <span className="save-status">
              {busy ? (
                <LoaderCircle size={14} className="spin" />
              ) : (
                <Check size={14} />
              )}{" "}
              {busy
                ? "Working…"
                : starting || recovering
                  ? "Connecting…"
                  : error
                    ? "Action needs attention"
                    : "Saved on this workstation"}
            </span>
            <button
              className="quiet"
              onClick={() => {
                setSettingsError("");
                setSettingsOpen(true);
              }}
              disabled={disabled || !session}
            >
              <Settings2 size={17} />
              Settings
            </button>
          </div>
        </header>
        <div className="workspace">
          <div className="heading-row">
            <div>
              <div className="eyebrow">RECONSTRUCT / REVIEW / REFINE</div>
              <h1>Motion, in alignment.</h1>
              <p>Inspect every view. Refine the joints. Reconstruct in 3D.</p>
            </div>
            <div className="heading-actions">
              <button onClick={() => setImportOpen(true)} disabled={disabled}>
                <Upload size={16} />
                Import capture
              </button>
              <button
                className="primary"
                onClick={() => detect()}
                disabled={
                  disabled ||
                  !session ||
                  session.source === "synthetic" ||
                  !modelReady
                }
              >
                <ScanLine size={17} />
                Run pose estimation
              </button>
            </div>
          </div>
          {error && (
            <div className="alert error" role="alert">
              <AlertCircle size={18} />
              <span>{error}</span>
              {session && (
                <button disabled={busy || dragging} onClick={reloadSession}>
                  Reload saved session
                </button>
              )}
              <button onClick={() => setError("")} aria-label="Dismiss error">
                <X size={16} />
              </button>
            </div>
          )}
          {notice && (
            <div className="toast" role="status">
              <Check size={16} />
              {notice}
            </div>
          )}
          {activeJob && (
            <div className="job-bar">
              <LoaderCircle className="spin" size={17} />
              <span>
                {connectionLost
                  ? "Connection lost. Reconnecting to the saved job…"
                  : activeJob.message}
                <small>
                  {" "}
                  · {activeJob.completed_frames}/{activeJob.total_frames} frames
                  saved
                </small>
              </span>
              <progress value={activeJob.progress} max={100} />
              <b>{activeJob.progress}%</b>
              <button
                disabled={activeJob.cancel_requested || connectionLost}
                onClick={async () => {
                  try {
                    const next = await api<Job>(
                      `/jobs/${activeJob.id}/cancel`,
                      json("POST"),
                    );
                    if (next.session_id === sessionRef.current?.id)
                      setJob(next);
                  } catch (e) {
                    setError((e as Error).message);
                  }
                }}
              >
                {activeJob.cancel_requested ? "Stopping…" : "Cancel"}
              </button>
            </div>
          )}
          {!activeJob && job?.can_resume && (
            <div className="job-recovery" role="status">
              <AlertCircle size={20} />
              <div>
                <strong>Pose estimation {job.status}</strong>
                <p>
                  {job.completed_frames} frames saved · {job.remaining_frames}{" "}
                  frames remaining. Your manual labels are preserved.
                </p>
              </div>
              <button
                className="primary"
                onClick={resume}
                disabled={disabled || !modelReady}
              >
                Resume remaining frames
              </button>
            </div>
          )}
          {!session || !frame ? (
            <div className="loading-workspace">
              {starting ? (
                <LoaderCircle size={30} className="spin" />
              ) : (
                <AlertCircle size={30} />
              )}
              <h2>
                {starting
                  ? "Preparing the pose workspace"
                  : "The workspace could not be loaded"}
              </h2>
              <p>
                {error
                  ? "Check that the local server is running. Your saved projects are kept on disk."
                  : "Loading saved sessions and the interactive reference sequence…"}
              </p>
              {!starting && (
                <button className="primary" onClick={initialize}>
                  Retry connection
                </button>
              )}
            </div>
          ) : (
            <>
              {session.source === "synthetic" && (
                <div className="demo-banner">
                  <span className="demo-badge">DEMO</span>
                  <span>
                    Synthetic reach sequence. The right wrist in cam2 is
                    deliberately misaligned on frames 10–31.
                  </span>
                  <button
                    disabled={navigationLocked}
                    onClick={() => {
                      setPlaying(false);
                      setIndex(20);
                      setSelected(10);
                      setCameraB("cam2");
                    }}
                  >
                    Inspect frame 20 <ChevronRight size={14} />
                  </button>
                </div>
              )}
              <div className="workspace-toolbar">
                <div className="view-label">
                  <Grid2X2 size={16} />
                  <strong>Multi-view reconstruction</strong>
                  <span className="subtle">
                    {names.length} calibrated cameras
                  </span>
                </div>
                <div className="toggles">
                  <label>
                    <input
                      type="checkbox"
                      checked={raw}
                      onChange={(e) => setRaw(e.target.checked)}
                    />
                    Raw predictions
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={projection}
                      onChange={(e) => setProjection(e.target.checked)}
                    />
                    Reprojection
                  </label>
                </div>
              </div>
              <p className="calibration-status">
                <strong>
                  {session.calibration.quality.source === "synthetic" ||
                  session.source === "synthetic"
                    ? "Synthetic camera geometry"
                    : session.calibration.quality.accepted === true
                      ? "Calibration fit accepted"
                      : "Imported calibration · fit not verified"}
                </strong>
                {session.calibration.quality.stereo_rms_px !== undefined &&
                  ` · Board RMS ${session.calibration.quality.stereo_rms_px.toFixed(2)} px`}
                {session.source !== "synthetic" &&
                  " · Verify physical scale and camera alignment with your rig."}
              </p>
              <div className="view-grid">
                <CameraView
                  name={cameraA}
                  available={names}
                  onCamera={(name) => {
                    if (!navigationLocked) setCameraA(name);
                  }}
                  view={frame.views[cameraA]}
                  frame={frame}
                  session={session}
                  selected={selected}
                  onSelect={setSelected}
                  onEdit={edit}
                  onInteraction={(active) => {
                    setDragging(active);
                    if (active) setPlaying(false);
                  }}
                  onActor={actor}
                  disabled={disabled || playing}
                  showRaw={raw}
                  showProjection={projection}
                  place={place}
                />
                <CameraView
                  name={cameraB}
                  available={names}
                  onCamera={(name) => {
                    if (!navigationLocked) setCameraB(name);
                  }}
                  view={frame.views[cameraB]}
                  frame={frame}
                  session={session}
                  selected={selected}
                  onSelect={setSelected}
                  onEdit={edit}
                  onInteraction={(active) => {
                    setDragging(active);
                    if (active) setPlaying(false);
                  }}
                  onActor={actor}
                  disabled={disabled || playing}
                  showRaw={raw}
                  showProjection={projection}
                  place={place}
                />
                <section className="skeleton-panel">
                  <header>
                    <div className="panel-title">
                      <Box size={16} />
                      3D skeleton
                    </div>
                    <span className="live-tag">
                      {frame.quality.valid}/17 joints
                    </span>
                  </header>
                  <Skeleton3D
                    session={session}
                    frame={frame}
                    selected={selected}
                    onSelect={setSelected}
                  />
                  <footer>
                    <span>
                      <span className="legend-dot green" />
                      Observed <span className="legend-dot amber" />
                      Corrected
                    </span>
                    <span>
                      {frame.quality.mean_error_px === null
                        ? "—"
                        : frame.quality.mean_error_px.toFixed(2)}{" "}
                      px mean error
                    </span>
                  </footer>
                </section>
              </div>
              <div className="edit-toolbar">
                <div className="segmented">
                  <button
                    className={!place ? "selected" : ""}
                    onClick={() => setPlace(false)}
                  >
                    <MousePointer2 size={15} />
                    Drag joints
                  </button>
                  <button
                    className={place ? "selected" : ""}
                    onClick={() => setPlace(true)}
                  >
                    <Crosshair size={15} />
                    Place missing joint
                  </button>
                </div>
                <label className="joint-picker">
                  Joint{" "}
                  <select
                    aria-label="Selected joint"
                    value={selected}
                    onChange={(e) => setSelected(Number(e.target.value))}
                  >
                    {session.joint_names.map((n, i) => (
                      <option key={n} value={i}>
                        {jointLabel(n)}
                      </option>
                    ))}
                  </select>
                  <ChevronDown size={13} />
                </label>
                <button
                  className="undo-button"
                  onClick={undo}
                  disabled={disabled || !session.history.length}
                >
                  <Undo2 size={16} />
                  Undo
                </button>
                <span className="edit-hint">
                  Drag a joint to correct its position. 3D updates on release.
                </span>
              </div>
              <div className="bottom-grid">
                <section className="timeline-panel">
                  <header>
                    <div>
                      <h2>Sequence review</h2>
                      <span>
                        {session.frames.length} frames ·{" "}
                        {session.frames.at(-1)!.timestamp.toFixed(2)}s span
                      </span>
                    </div>
                    <button
                      className="review-next"
                      onClick={nextReview}
                      disabled={navigationLocked}
                    >
                      {reviewCount ? (
                        <>
                          <AlertCircle size={14} />
                          {reviewCount} to review
                        </>
                      ) : (
                        <>
                          <Check size={14} />
                          All frames covered
                        </>
                      )}
                      <ChevronRight size={14} />
                    </button>
                  </header>
                  <div
                    className="timeline-strip"
                    aria-label="Frame quality timeline"
                  >
                    {session.frames.map((f, i) => (
                      <button
                        key={f.id}
                        disabled={navigationLocked}
                        aria-label={`Go to frame ${f.id}${f.quality.needs_review ? ", needs review" : ""}`}
                        title={`Frame ${f.id} · ${f.quality.valid}/17 joints`}
                        className={
                          (i === index ? "current " : "") +
                          (f.quality.corrected
                            ? "corrected"
                            : f.quality.needs_review
                              ? "review"
                              : "good")
                        }
                        onClick={() => {
                          setIndex(i);
                          setPlaying(false);
                        }}
                      >
                        <span
                          style={{
                            height: `${Math.max(12, (f.quality.valid / 17) * 100)}%`,
                          }}
                        />
                      </button>
                    ))}
                  </div>
                  <input
                    className="timeline-slider"
                    disabled={navigationLocked}
                    aria-label="Current frame"
                    type="range"
                    min={0}
                    max={session.frames.length - 1}
                    value={index}
                    onChange={(e) => {
                      setIndex(Number(e.target.value));
                      setPlaying(false);
                    }}
                  />
                  <div className="timeline-controls">
                    <div className="transport">
                      <button
                        title="First frame"
                        disabled={navigationLocked}
                        aria-label="First frame"
                        onClick={() => {
                          setIndex(0);
                          setPlaying(false);
                        }}
                      >
                        <SkipBack size={16} />
                      </button>
                      <button
                        title="Previous frame"
                        aria-label="Previous frame"
                        disabled={navigationLocked || index === 0}
                        onClick={() => {
                          setIndex((i) => Math.max(0, i - 1));
                          setPlaying(false);
                        }}
                      >
                        <ChevronLeft size={18} />
                      </button>
                      <button
                        className="play-button"
                        disabled={navigationLocked || disabled}
                        aria-label={playing ? "Pause" : "Play sequence"}
                        onClick={() => {
                          if (index === session.frames.length - 1) setIndex(0);
                          setPlaying(!playing);
                        }}
                      >
                        {playing ? <Pause size={17} /> : <Play size={17} />}
                      </button>
                      <button
                        title="Next frame"
                        aria-label="Next frame"
                        disabled={
                          navigationLocked ||
                          index === session.frames.length - 1
                        }
                        onClick={() => {
                          setIndex((i) =>
                            Math.min(session.frames.length - 1, i + 1),
                          );
                          setPlaying(false);
                        }}
                      >
                        <ChevronRight size={18} />
                      </button>
                    </div>
                    <div className="frame-readout">
                      FRAME <b>{String(frame.id).padStart(4, "0")}</b>
                      <span>{frame.timestamp.toFixed(3)} s</span>
                    </div>
                    <div className="timeline-legend">
                      <span>
                        <i className="legend-dot green" />
                        Observed
                      </span>
                      <span>
                        <i className="legend-dot red" />
                        Review
                      </span>
                      <span>
                        <i className="legend-dot amber" />
                        Corrected
                      </span>
                    </div>
                  </div>
                </section>
                <section className="inspector">
                  <header>
                    <h2>{jointLabel(session.joint_names[selected])}</h2>
                    <span className={"status-chip " + joint?.status}>
                      {joint?.status}
                    </span>
                  </header>
                  <div className="position-values">
                    {["X", "Y", "Z"].map((axis, i) => (
                      <div key={axis}>
                        <span>{axis}</span>
                        <b>{joint?.point ? joint.point[i].toFixed(3) : "—"}</b>
                        <small>m</small>
                      </div>
                    ))}
                  </div>
                  <div className="inspector-row">
                    <span>Reprojection error</span>
                    <b>
                      {joint?.error_px === null
                        ? "—"
                        : joint?.error_px.toFixed(2) + " px"}
                    </b>
                  </div>
                  <div className="inspector-row">
                    <span>Accepted views</span>
                    <b>{joint?.cameras.join(", ") || "None"}</b>
                  </div>
                  {!!joint?.excluded.length && (
                    <p className="inspector-note">
                      Excluded: {joint.excluded.join(", ")}. Correct its joint
                      to align this view.
                    </p>
                  )}
                  {joint?.reason && (
                    <p className="inspector-note">{joint.reason}</p>
                  )}
                </section>
              </div>
              <div className="workspace-footer">
                <span>
                  <Check size={14} />
                  Manual labels persist across detection runs.
                </span>
                <div>
                  <button
                    className="text-button"
                    onClick={() => detect(true)}
                    disabled={
                      disabled || session.source === "synthetic" || !modelReady
                    }
                  >
                    Re-detect current frame
                  </button>
                  <a
                    className="button"
                    href={`/api/sessions/${session.id}/project`}
                  >
                    <Layers size={15} />
                    Save project ZIP
                  </a>
                  <a
                    className="button primary"
                    href={`/api/sessions/${session.id}/export`}
                  >
                    <Download size={16} />
                    Export 3D motion
                  </a>
                </div>
              </div>
            </>
          )}
        </div>
      </main>
      {importOpen && (
        <ImportDialog
          onClose={() => setImportOpen(false)}
          onImported={(s) => {
            load(s);
            setImportOpen(false);
            setNotice("Capture loaded. Run pose estimation to detect joints.");
          }}
          calibration={calibration}
        />
      )}
      {calibrateOpen && (
        <CalibrationDialog
          onClose={() => setCalibrateOpen(false)}
          onBuilt={built}
        />
      )}
      {sessionsOpen && (
        <Modal
          title="Saved sessions"
          onClose={() => !busy && setSessionsOpen(false)}
          busy={busy}
        >
          <div className="sessions-filter">
            <label>
              <input
                type="checkbox"
                checked={archivedList}
                disabled={busy}
                onChange={(e) => {
                  setArchivedList(e.target.checked);
                  setRenaming(null);
                  void refreshSaved(e.target.checked);
                }}
              />{" "}
              Show archived sessions
            </label>
            <button disabled={busy} onClick={() => refreshSaved()}>
              Refresh
            </button>
          </div>
          {sessionError && (
            <p role="alert" className="error">
              {sessionError}
            </p>
          )}
          <div className="sessions-list">
            {!saved.length && (
              <p className="help">
                {archivedList
                  ? "No archived sessions. Archiving keeps all images and corrections."
                  : "No saved captures yet. Import a capture or open a demo."}
              </p>
            )}
            {saved.map((s) => (
              <div className="session-entry" key={s.id}>
                <button
                  disabled={busy}
                  onClick={async () => {
                    if (!lock()) return;
                    try {
                      load(await api<Session>(`/sessions/${s.id}`));
                      setSessionsOpen(false);
                    } catch (e) {
                      setSessionError((e as Error).message);
                    } finally {
                      unlock();
                    }
                  }}
                >
                  <FolderOpen size={21} />
                  <span>
                    <b>{s.name}</b>
                    <small>
                      {s.frame_count} frames ·{" "}
                      {s.source === "synthetic"
                        ? "Synthetic demo"
                        : "Camera capture"}
                      {s.archived ? " · Archived" : ""}
                    </small>
                  </span>
                  <ChevronRight size={18} />
                </button>
                {renaming === s.id ? (
                  <form
                    className="session-rename"
                    onSubmit={(e) => {
                      e.preventDefault();
                      void manageSession(s, "rename");
                    }}
                  >
                    <input
                      autoFocus
                      aria-label="New session name"
                      value={sessionName}
                      maxLength={120}
                      required
                      onChange={(e) => setSessionName(e.target.value)}
                    />
                    <button
                      className="primary"
                      disabled={busy || !sessionName.trim()}
                    >
                      Save name
                    </button>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => setRenaming(null)}
                    >
                      Cancel
                    </button>
                  </form>
                ) : (
                  <div className="session-entry-actions">
                    <button
                      disabled={busy}
                      onClick={() => {
                        setRenaming(s.id);
                        setSessionName(s.name);
                      }}
                    >
                      Rename
                    </button>
                    <button
                      disabled={busy}
                      onClick={() =>
                        manageSession(s, s.archived ? "unarchive" : "archive")
                      }
                    >
                      {s.archived ? "Restore to workspace" : "Archive"}
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
          <div className="dialog-actions">
            <label className="button file-button">
              <Upload size={15} />
              Restore project ZIP
              <input
                aria-label="Restore project ZIP"
                type="file"
                accept=".zip"
                onChange={restore}
                disabled={busy}
              />
            </label>
            <button
              disabled={disabled}
              onClick={() => {
                setSessionsOpen(false);
                freshDemo();
              }}
            >
              New synthetic demo
            </button>
          </div>
        </Modal>
      )}
      {settingsOpen && session && (
        <Modal
          title="Reconstruction settings"
          onClose={() => !busy && setSettingsOpen(false)}
          busy={busy}
        >
          <form
            onSubmit={async (e) => {
              e.preventDefault();
              const data = new FormData(e.currentTarget);
              if (!lock()) return;
              setSettingsError("");
              const token = epoch.current;
              try {
                accept(
                  await api<Session>(
                    `/sessions/${session.id}/settings`,
                    json("PATCH", {
                      ...Object.fromEntries(
                        [...data].map(([k, v]) => [k, Number(v)]),
                      ),
                      revision: session.revision,
                    }),
                  ),
                  token,
                );
                setSettingsOpen(false);
                setNotice("Geometry checks updated for the whole sequence.");
              } catch (e) {
                setSettingsError((e as Error).message);
              } finally {
                unlock();
              }
            }}
          >
            <p className="dialog-intro">
              Geometry is evaluated per joint. Manual labels are authoritative,
              but inconsistent or poorly conditioned views remain rejected.
            </p>
            <label className="field">
              Minimum detector confidence
              <input
                name="min_confidence"
                type="number"
                min={0}
                max={1}
                step={0.05}
                defaultValue={session.settings.min_confidence}
              />
            </label>
            <label className="field">
              Maximum reprojection error (px)
              <input
                name="max_error_px"
                type="number"
                min={0.1}
                max={30}
                step={0.1}
                defaultValue={session.settings.max_error_px}
              />
            </label>
            <label className="field">
              Minimum triangulation angle (degrees)
              <input
                name="min_angle_deg"
                type="number"
                min={0.1}
                max={30}
                step={0.1}
                defaultValue={session.settings.min_angle_deg}
              />
            </label>
            <p className="help">
              No smoothing or gap filling is applied. Unobserved joints stay
              missing in the export.
            </p>
            {settingsError && (
              <p className="error" role="alert">
                {settingsError}
              </p>
            )}
            <div className="dialog-actions">
              <button
                type="button"
                onClick={() =>
                  downloadJSON(session.calibration, "camera_calibration.json")
                }
              >
                Download calibration
              </button>
              <button className="primary" disabled={busy}>
                <SlidersHorizontal size={16} />
                Apply to sequence
              </button>
            </div>
          </form>
        </Modal>
      )}
    </div>
  );
}
