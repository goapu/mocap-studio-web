export type Observation = {
  xy: [number, number] | null;
  confidence: number;
  manual?: boolean;
};
export type Joint = {
  point: [number, number, number] | null;
  status: string;
  error_px: number | null;
  cameras: string[];
  excluded: string[];
  angle_deg: number | null;
  reason: string | null;
};
export type View = {
  image: string;
  width: number;
  height: number;
  raw: Observation[];
  edits: Record<string, [number, number] | null>;
  effective: Observation[];
  reprojected: ([number, number] | null)[];
  candidates: Observation[][];
  actor_index: number | null;
};
export type Frame = {
  id: number;
  timestamp: number;
  views: Record<string, View>;
  pose: Joint[];
  quality: {
    valid: number;
    corrected: number;
    mean_error_px: number | null;
    needs_review: boolean;
  };
};
export type Calibration = {
  units: string;
  world_frame: "camera" | "z_up";
  cameras: Record<
    string,
    {
      K: number[][];
      dist: number[];
      R: number[][];
      T: number[];
      image_size: number[];
    }
  >;
  quality: { source: string; accepted: boolean | null; stereo_rms_px?: number };
};
export type Session = {
  id: string;
  name: string;
  source: string;
  fps: number;
  revision: number;
  frames: Frame[];
  joint_names: string[];
  bones: [number, number][];
  calibration: Calibration;
  settings: {
    min_confidence: number;
    max_error_px: number;
    min_angle_deg: number;
  };
  history: unknown[];
};
export type Summary = {
  id: string;
  name: string;
  source: string;
  frame_count: number;
  revision: number;
  archived?: boolean;
};
export type Job = {
  id: string;
  session_id: string;
  status: string;
  progress: number;
  message: string;
  total_frames: number;
  completed_frames: number;
  remaining_frames: number;
  cancel_requested: boolean;
  can_resume: boolean;
  created_at: string;
  updated_at: string;
};
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch("/api" + path, init);
  } catch {
    throw new Error(
      "Cannot reach the local server. Check that Mocap Studio is running and try again.",
    );
  }
  if (!res.ok) {
    const data = await res
      .json()
      .catch(() => ({ detail: "The server could not complete this request." }));
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail),
    );
  }
  return res.json();
}
export function json(method: string, data?: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: data === undefined ? undefined : JSON.stringify(data),
  };
}
export const jointLabel = (name: string) =>
  name.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
