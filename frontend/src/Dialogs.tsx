import { useEffect, useRef, useState } from "react";
import { X, Upload, Plus, LoaderCircle, Check } from "lucide-react";
import { api } from "./types";
import type { Session, Calibration } from "./types";

export function Modal({
  title,
  onClose,
  children,
  busy = false,
}: {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
  busy?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    ref.current?.showModal();
  }, []);
  return (
    <dialog
      aria-label={title}
      aria-busy={busy}
      ref={ref}
      onCancel={(e) => {
        e.preventDefault();
        if (!busy) onClose();
      }}
    >
      <header>
        <h2>{title}</h2>
        <button
          type="button"
          aria-label="Close dialog"
          onClick={onClose}
          disabled={busy}
        >
          <X size={20} />
        </button>
      </header>
      {children}
    </dialog>
  );
}

export function ImportDialog({
  onClose,
  onImported,
  calibration,
}: {
  onClose: () => void;
  onImported: (s: Session) => void;
  calibration: File | null;
}) {
  const [count, setCount] = useState(2),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const [selected, setSelected] = useState<Record<string, File[]>>({});
  const [cal, setCal] = useState<File | null>(calibration);
  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError("");
    setBusy(true);
    const data = new FormData(e.currentTarget);
    Object.entries(selected).forEach(([key, files]) =>
      files.forEach((f) => data.append(key, f)),
    );
    if (cal) data.append("calibration", cal);
    try {
      onImported(
        await api<Session>("/sessions", { method: "POST", body: data }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Modal
      title="Import synchronized capture"
      busy={busy}
      onClose={() => !busy && onClose()}
    >
      <form onSubmit={submit}>
        <p className="dialog-intro">
          Add a video or numbered images for each camera. Use aligned start
          times, matching video FPS, and the resolution used for calibration.
        </p>
        <label className="field">
          Session name
          <input
            name="name"
            required
            defaultValue="Movement study"
            maxLength={100}
          />
        </label>
        <div className="capture-inputs">
          {Array.from({ length: count }, (_, i) => `cam${i + 1}`).map(
            (name) => (
              <label
                key={name}
                className={
                  "upload-row " + (selected[name]?.length ? "has-files" : "")
                }
              >
                <span className="camera-pill">{name}</span>
                <div>
                  <b>
                    {selected[name]?.length
                      ? `${selected[name].length} file${selected[name].length === 1 ? "" : "s"} selected`
                      : "Choose video or image sequence"}
                  </b>
                  <small>
                    {selected[name]?.[0]?.name || "MP4, MOV, AVI, JPG or PNG"}
                  </small>
                </div>
                <Upload size={18} />
                <input
                  aria-label={`Upload ${name} capture`}
                  type="file"
                  multiple
                  accept="video/*,image/*"
                  disabled={busy}
                  onChange={(e) =>
                    setSelected({
                      ...selected,
                      [name]: Array.from(e.target.files || []),
                    })
                  }
                />
              </label>
            ),
          )}
        </div>
        {count < 6 && (
          <button
            type="button"
            className="text-button"
            disabled={busy}
            onClick={() => setCount(count + 1)}
          >
            <Plus size={15} />
            Add another camera
          </button>
        )}
        <label className="upload-row calibration-upload">
          <span>
            <Upload size={20} />
          </span>
          <div>
            <b>{cal ? "Calibration attached" : "Attach camera calibration"}</b>
            <small>{cal?.name || "JSON or compatible MocapStudio NPZ"}</small>
          </div>
          <input
            aria-label="Upload calibration"
            type="file"
            accept=".json,.npz"
            disabled={busy}
            onChange={(e) => setCal(e.target.files?.[0] || null)}
          />
          {cal && <Check size={18} />}
        </label>
        <div className="field-row">
          <label className="field">
            Image sequence FPS
            <input
              name="fps"
              type="number"
              defaultValue={30}
              min={1}
              max={240}
              step=".01"
            />
          </label>
          <label className="field">
            Video frame stride
            <input
              name="stride"
              type="number"
              defaultValue={1}
              min={1}
              max={60}
            />
          </label>
        </div>
        <p className="help">
          Videos use their recorded FPS. Frame stride samples every Nth video
          frame while preserving timestamps. Maximum 300 sampled frames and 500
          MB. Images need shared numbers, such as frame001.jpg.
        </p>
        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}
        <div className="dialog-actions">
          <button type="button" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button
            className="primary"
            disabled={
              busy || !cal || !selected.cam1?.length || !selected.cam2?.length
            }
          >
            {busy ? (
              <LoaderCircle className="spin" size={17} />
            ) : (
              <Upload size={17} />
            )}{" "}
            {busy ? "Preparing frames…" : "Open capture"}
          </button>
        </div>
      </form>
    </Modal>
  );
}

export function CalibrationDialog({
  onClose,
  onBuilt,
}: {
  onClose: () => void;
  onBuilt: (cal: Calibration) => void;
}) {
  const [count, setCount] = useState(2),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const [selected, setSelected] = useState<Record<string, File[]>>({});
  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setBusy(true);
    setError("");
    const data = new FormData(e.currentTarget);
    Object.entries(selected).forEach(([key, files]) =>
      files.forEach((f) => data.append(key, f)),
    );
    try {
      onBuilt(
        await api<Calibration>("/calibrate", { method: "POST", body: data }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Modal
      title="Calibrate your cameras"
      busy={busy}
      onClose={() => !busy && onClose()}
    >
      <form onSubmit={submit}>
        <p className="dialog-intro">
          Use a chessboard captured simultaneously by every camera. Add at least
          12 matching, numbered images per camera with varied board angles and
          positions.
        </p>
        <div className="field-row">
          <label className="field">
            Inner columns
            <input
              name="cols"
              type="number"
              defaultValue={9}
              min={3}
              max={20}
            />
          </label>
          <label className="field">
            Inner rows
            <input
              name="rows"
              type="number"
              defaultValue={6}
              min={3}
              max={20}
            />
          </label>
          <label className="field">
            Square size (m)
            <input
              name="square"
              type="number"
              defaultValue={0.025}
              min={0.001}
              max={1}
              step={0.001}
            />
          </label>
        </div>
        {Array.from({ length: count }, (_, i) => `cam${i + 1}`).map((name) => (
          <label key={name} className="upload-row">
            <span className="camera-pill">{name}</span>
            <div>
              <b>
                {selected[name]?.length
                  ? `${selected[name].length} board images`
                  : "Add calibration images"}
              </b>
              <small>frame001.jpg, frame002.jpg…</small>
            </div>
            <Upload size={17} />
            <input
              aria-label={`Upload ${name} calibration images`}
              type="file"
              accept="image/*"
              multiple
              disabled={busy}
              onChange={(e) =>
                setSelected({
                  ...selected,
                  [name]: Array.from(e.target.files || []),
                })
              }
            />
          </label>
        ))}
        {count < 6 && (
          <button
            type="button"
            className="text-button"
            disabled={busy}
            onClick={() => setCount(count + 1)}
          >
            <Plus size={15} />
            Add camera
          </button>
        )}
        <p className="help">
          Calibration is accepted only when stereo RMS is at most 1.5 px. This
          verifies the board fit; it does not by itself certify body-pose
          accuracy.
        </p>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        <div className="dialog-actions">
          <button type="button" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button
            className="primary"
            disabled={busy || !selected.cam1?.length || !selected.cam2?.length}
          >
            {busy && <LoaderCircle size={17} className="spin" />}
            {busy ? "Finding corners and calibrating…" : "Build calibration"}
          </button>
        </div>
      </form>
    </Modal>
  );
}
