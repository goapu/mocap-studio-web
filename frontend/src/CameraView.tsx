import { useEffect, useRef, useState } from "react";
import {
  Crosshair,
  Camera,
  RotateCcw,
  EyeOff,
  ZoomIn,
  ZoomOut,
  Focus,
} from "lucide-react";
import type { Frame, Session, View } from "./types";
import { jointLabel } from "./types";

type Props = {
  name: string;
  available: string[];
  onCamera: (name: string) => void;
  view?: View;
  frame: Frame;
  session: Session;
  selected: number;
  onSelect: (i: number) => void;
  onInteraction: (active: boolean) => void;
  onEdit: (
    frameId: number,
    name: string,
    joint: number,
    xy: [number, number] | null,
    reset?: boolean,
  ) => void;
  onActor: (name: string, index: number) => void;
  disabled: boolean;
  showRaw: boolean;
  showProjection: boolean;
  place: boolean;
};
export default function CameraView(p: Props) {
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<{
    joint: number;
    start: [number, number];
    moved: boolean;
    frameId: number;
    sessionId: string;
    camera: string;
  } | null>(null);
  const [draft, setDraft] = useState<{
    joint: number;
    xy: [number, number];
  } | null>(null);
  const [crop, setCrop] = useState<[number, number, number, number] | null>(
    null,
  );
  useEffect(() => {
    setCrop(null);
    setDraft(null);
    drag.current = null;
    p.onInteraction(false);
  }, [p.name, p.session.id, p.frame.id]);
  if (!p.view)
    return (
      <section className="camera-panel">
        <header>
          <Camera size={16} />
          <select
            aria-label={`Camera panel ${p.name}`}
            value={p.name}
            onChange={(e) => p.onCamera(e.target.value)}
          >
            {p.available.map((name) => (
              <option key={name}>{name}</option>
            ))}
          </select>
        </header>
        <div className="missing-view">
          No image for this camera at this timestamp.
        </div>
      </section>
    );
  const v = p.view;
  const coordinate = (e: React.PointerEvent): [number, number] => {
    const pt = svg.current!.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const x = pt.matrixTransform(svg.current!.getScreenCTM()!.inverse());
    return [
      Math.max(0, Math.min(v.width - 1, x.x)),
      Math.max(0, Math.min(v.height - 1, x.y)),
    ];
  };
  const points = v.effective.map((o, i) =>
    draft?.joint === i ? draft.xy : o.xy,
  );
  function start(e: React.PointerEvent, i: number) {
    if (p.disabled) return;
    e.stopPropagation();
    p.onSelect(i);
    const xy = coordinate(e);
    p.onInteraction(true);
    drag.current = {
      joint: i,
      start: xy,
      moved: false,
      frameId: p.frame.id,
      sessionId: p.session.id,
      camera: p.name,
    };
    svg.current?.setPointerCapture(e.pointerId);
  }
  function move(e: React.PointerEvent) {
    if (!drag.current) return;
    const xy = coordinate(e);
    if (
      Math.hypot(xy[0] - drag.current.start[0], xy[1] - drag.current.start[1]) >
      1
    )
      drag.current.moved = true;
    setDraft({ joint: drag.current.joint, xy });
  }
  function end(e: React.PointerEvent) {
    if (!drag.current) return;
    const d = drag.current;
    drag.current = null;
    if (
      d.moved &&
      d.frameId === p.frame.id &&
      d.sessionId === p.session.id &&
      d.camera === p.name
    )
      p.onEdit(d.frameId, p.name, d.joint, coordinate(e));
    p.onInteraction(false);
    setDraft(null);
  }
  const selected = v.effective[p.selected];
  const full: [number, number, number, number] = [0, 0, v.width, v.height];
  const bounds = crop || full;
  function zoom(factor: number) {
    const [x, y, w, h] = bounds;
    const target = points[p.selected] || [x + w / 2, y + h / 2];
    const width = Math.min(v.width, Math.max(v.width / 8, w * factor)),
      height = (width * v.height) / v.width;
    setCrop([
      Math.max(0, Math.min(v.width - width, target[0] - width / 2)),
      Math.max(0, Math.min(v.height - height, target[1] - height / 2)),
      width,
      height,
    ]);
  }
  return (
    <section className="camera-panel">
      <header>
        <div className="panel-title">
          <Camera size={15} />
          <select
            aria-label={`Camera panel ${p.name}`}
            value={p.name}
            onChange={(e) => p.onCamera(e.target.value)}
          >
            {p.available.map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </div>
        <span className="dimension">
          {v.width} × {v.height}
        </span>
      </header>
      <div className={"image-stage " + (p.place ? "placing" : "")}>
        <svg
          ref={svg}
          viewBox={bounds.join(" ")}
          aria-label={`${p.name} pose editor`}
          onPointerMove={move}
          onPointerUp={end}
          onPointerCancel={() => {
            drag.current = null;
            setDraft(null);
            p.onInteraction(false);
          }}
          onPointerDown={(e) => {
            if (p.place && !p.disabled)
              p.onEdit(p.frame.id, p.name, p.selected, coordinate(e));
          }}
        >
          <image
            href={`/api/sessions/${p.session.id}/media/${v.image}`}
            width={v.width}
            height={v.height}
          />
          {v.actor_index === null &&
            v.candidates.length > 1 &&
            v.candidates.map((candidate, k) => (
              <g key={k} opacity=".65">
                {p.session.bones.map(([a, b]) =>
                  candidate[a].xy && candidate[b].xy ? (
                    <line
                      key={`${a}-${b}`}
                      x1={candidate[a].xy![0]}
                      y1={candidate[a].xy![1]}
                      x2={candidate[b].xy![0]}
                      y2={candidate[b].xy![1]}
                      stroke={["#bbf46c", "#eeb4ff", "#ffcb71"][k % 3]}
                      strokeWidth="5"
                    />
                  ) : null,
                )}
                {candidate[0].xy && (
                  <text
                    x={candidate[0].xy[0] + 12}
                    y={candidate[0].xy[1] - 18}
                    fill="white"
                    fontSize="25"
                  >
                    Person {k + 1}
                  </text>
                )}
              </g>
            ))}
          {p.showRaw &&
            v.raw.map(
              (o, i) =>
                o.xy && (
                  <circle
                    key={i}
                    cx={o.xy[0]}
                    cy={o.xy[1]}
                    r="7"
                    fill="none"
                    stroke="#9ba6af"
                    strokeWidth="2"
                    opacity=".7"
                    pointerEvents="none"
                  />
                ),
            )}
          {p.session.bones.map(([a, b]) =>
            points[a] && points[b] ? (
              <line
                key={`${a}-${b}`}
                x1={points[a]![0]}
                y1={points[a]![1]}
                x2={points[b]![0]}
                y2={points[b]![1]}
                stroke={a % 2 === 1 ? "#65d4ec" : "#c4ee95"}
                strokeWidth="4"
                opacity=".9"
                pointerEvents="none"
              />
            ) : null,
          )}
          {points[0] && points[5] && points[6] && (
            <line
              x1={points[0][0]}
              y1={points[0][1]}
              x2={(points[5][0] + points[6][0]) / 2}
              y2={(points[5][1] + points[6][1]) / 2}
              stroke="#c4ee95"
              strokeWidth="4"
              pointerEvents="none"
            />
          )}
          {p.showProjection &&
            v.reprojected.map(
              (xy, i) =>
                xy && (
                  <g key={i} pointerEvents="none">
                    {points[i] && (
                      <line
                        x1={points[i]![0]}
                        y1={points[i]![1]}
                        x2={xy[0]}
                        y2={xy[1]}
                        stroke="#ffa963"
                        strokeDasharray="5 5"
                        strokeWidth="2"
                      />
                    )}
                    <path
                      d={`M${xy[0] - 5},${xy[1]}h10 M${xy[0]},${xy[1] - 5}v10`}
                      stroke="#ffa963"
                      strokeWidth="2"
                    />
                  </g>
                ),
            )}
          {points.map(
            (xy, i) =>
              xy && (
                <g
                  key={i}
                  onPointerDown={(e) => start(e, i)}
                  className="joint-handle"
                  role="button"
                  tabIndex={p.disabled ? -1 : 0}
                  aria-pressed={i === p.selected}
                  aria-disabled={p.disabled}
                  aria-label={`${jointLabel(p.session.joint_names[i])} in ${p.name}. Arrow keys move one pixel; Shift moves ten.`}
                  onFocus={() => p.onSelect(i)}
                  onKeyDown={(e) => {
                    if (p.disabled) return;
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      p.onSelect(i);
                      return;
                    }
                    const delta: Record<string, [number, number]> = {
                      ArrowLeft: [-1, 0],
                      ArrowRight: [1, 0],
                      ArrowUp: [0, -1],
                      ArrowDown: [0, 1],
                    };
                    const direction = delta[e.key];
                    if (direction) {
                      e.preventDefault();
                      e.stopPropagation();
                      const step = e.shiftKey ? 10 : 1;
                      p.onEdit(p.frame.id, p.name, i, [
                        Math.max(
                          0,
                          Math.min(v.width - 1, xy[0] + direction[0] * step),
                        ),
                        Math.max(
                          0,
                          Math.min(v.height - 1, xy[1] + direction[1] * step),
                        ),
                      ]);
                    }
                  }}
                >
                  <circle cx={xy[0]} cy={xy[1]} r="17" fill="transparent" />
                  {i === p.selected && (
                    <circle
                      cx={xy[0]}
                      cy={xy[1]}
                      r="15"
                      fill="none"
                      stroke="white"
                      strokeWidth="2"
                    />
                  )}
                  <circle
                    cx={xy[0]}
                    cy={xy[1]}
                    r={i === p.selected ? 7 : 5}
                    stroke="#172020"
                    strokeWidth="2"
                    fill={
                      v.edits[String(i)] !== undefined
                        ? "#ffc778"
                        : v.effective[i].confidence <
                            p.session.settings.min_confidence
                          ? "#fa807a"
                          : i % 2 === 1
                            ? "#65d4ec"
                            : "#c4ee95"
                    }
                  />
                </g>
              ),
          )}
        </svg>
        <div className="camera-zoom">
          <button
            aria-label={`Zoom in ${p.name}`}
            title="Zoom around selected joint"
            onClick={() => zoom(0.65)}
          >
            <ZoomIn size={14} />
          </button>
          <button
            aria-label={`Zoom out ${p.name}`}
            title="Zoom out"
            onClick={() => zoom(1 / 0.65)}
          >
            <ZoomOut size={14} />
          </button>
          <button
            aria-label={`Fit image ${p.name}`}
            title="Fit whole image"
            onClick={() => setCrop(null)}
          >
            <Focus size={14} />
          </button>
        </div>
        <span className="view-tag">{p.frame.timestamp.toFixed(3)}s</span>
        {p.place && (
          <span className="place-message">
            Click to place{" "}
            {jointLabel(p.session.joint_names[p.selected]).toLowerCase()}
          </span>
        )}
      </div>
      {v.candidates.length > 1 && (
        <div className="actor-select">
          <label>
            Match the same person{" "}
            <select
              aria-label={`Select person in ${p.name}`}
              value={v.actor_index ?? ""}
              disabled={p.disabled}
              onChange={(e) => {
                if (e.target.value !== "")
                  p.onActor(p.name, Number(e.target.value));
              }}
            >
              <option value="">Choose person…</option>
              {v.candidates.map((_, i) => (
                <option key={i} value={i}>
                  Person {i + 1}
                </option>
              ))}
            </select>
          </label>
        </div>
      )}
      <PixelEditor
        key={`${p.session.id}-${p.name}-${p.frame.id}-${p.selected}`}
        camera={p.name}
        xy={selected?.xy || null}
        width={v.width}
        height={v.height}
        disabled={p.disabled}
        onSave={(xy) => p.onEdit(p.frame.id, p.name, p.selected, xy)}
      />
      <footer>
        <span>
          <Crosshair size={14} />
          {selected?.manual
            ? "Manually labeled"
            : selected?.xy
              ? `${Math.round(selected.confidence * 100)}% confidence`
              : "Joint missing"}
        </span>
        <div>
          <button
            aria-label={`Mark selected joint missing in ${p.name}`}
            title="Mark joint missing"
            disabled={p.disabled}
            onClick={() => p.onEdit(p.frame.id, p.name, p.selected, null)}
          >
            <EyeOff size={15} />
          </button>
          <button
            aria-label={`Reset selected joint in ${p.name}`}
            title="Restore model prediction"
            disabled={p.disabled || !(String(p.selected) in v.edits)}
            onClick={() => p.onEdit(p.frame.id, p.name, p.selected, null, true)}
          >
            <RotateCcw size={15} />
          </button>
        </div>
      </footer>
    </section>
  );
}

function PixelEditor({
  camera,
  xy,
  width,
  height,
  disabled,
  onSave,
}: {
  camera: string;
  xy: [number, number] | null;
  width: number;
  height: number;
  disabled: boolean;
  onSave: (xy: [number, number]) => void;
}) {
  const [x, setX] = useState(xy ? xy[0].toFixed(1) : "");
  const [y, setY] = useState(xy ? xy[1].toFixed(1) : "");
  useEffect(() => {
    setX(xy ? xy[0].toFixed(1) : "");
    setY(xy ? xy[1].toFixed(1) : "");
  }, [xy?.[0], xy?.[1]]);
  return (
    <form
      className="pixel-editor"
      onSubmit={(e) => {
        e.preventDefault();
        if (!disabled && x !== "" && y !== "") onSave([Number(x), Number(y)]);
      }}
    >
      <span>Position · px</span>
      <label>
        X
        <input
          aria-label={`Joint X in ${camera}`}
          type="number"
          required
          min={0}
          max={width - 1}
          step="any"
          value={x}
          onChange={(e) => setX(e.target.value)}
          disabled={disabled}
        />
      </label>
      <label>
        Y
        <input
          aria-label={`Joint Y in ${camera}`}
          type="number"
          required
          min={0}
          max={height - 1}
          step="any"
          value={y}
          onChange={(e) => setY(e.target.value)}
          disabled={disabled}
        />
      </label>
      <button
        title="Save precise joint position"
        aria-label={`Apply joint position in ${camera}`}
        disabled={disabled || x === "" || y === ""}
      >
        Apply
      </button>
    </form>
  );
}
