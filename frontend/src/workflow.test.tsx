// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import App from "./App";
import CameraView from "./CameraView";
import type { Session, Job } from "./types";
const mock = vi.hoisted(() => ({ api: vi.fn() }));
vi.mock("./types", async (original) => ({
  ...(await original<typeof import("./types")>()),
  api: mock.api,
}));
vi.mock("./Skeleton3D", () => ({ default: () => <div>3D viewer</div> }));
function capture(): Session {
  const observations = Array.from({ length: 17 }, () => ({
    xy: [100, 200] as [number, number],
    confidence: 0.9,
  }));
  const view = {
    image: "cam1_000000.jpg",
    width: 640,
    height: 480,
    raw: observations,
    effective: observations,
    edits: {},
    reprojected: Array(17).fill(null),
    candidates: [],
    actor_index: null,
  };
  return {
    id: "session1",
    name: "Walking capture",
    source: "capture",
    fps: 30,
    revision: 7,
    history: [],
    joint_names: Array.from({ length: 17 }, (_, i) =>
      i === 10 ? "right_wrist" : `joint_${i}`,
    ),
    bones: [[5, 6]],
    settings: { min_confidence: 0.35, max_error_px: 6, min_angle_deg: 1 },
    calibration: {
      units: "meters",
      world_frame: "camera",
      quality: { source: "import", accepted: null },
      cameras: Object.fromEntries(
        ["cam1", "cam2"].map((name) => [
          name,
          {
            K: [
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
            ],
            dist: [],
            R: [
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
            ],
            T: [0, 0, 0],
            image_size: [640, 480],
          },
        ]),
      ),
    },
    frames: [0, 1].map((id) => ({
      id,
      timestamp: id / 30,
      views: { cam1: structuredClone(view), cam2: structuredClone(view) },
      pose: Array.from({ length: 17 }, () => ({
        point: [0, 0, 2] as [number, number, number],
        status: "observed",
        error_px: 0,
        cameras: ["cam1", "cam2"],
        excluded: [],
        angle_deg: 20,
        reason: null,
      })),
      quality: {
        valid: 17,
        corrected: 0,
        mean_error_px: 0,
        needs_review: false,
      },
    })),
  };
}
function job(status = "interrupted"): Job {
  return {
    id: "job1",
    session_id: "session1",
    status,
    progress: 50,
    message: "Processing interrupted",
    total_frames: 2,
    completed_frames: 1,
    remaining_frames: 1,
    cancel_requested: false,
    can_resume: status === "interrupted",
    created_at: "",
    updated_at: "",
  };
}
function apiDefaults(latest: Job | null = null) {
  mock.api.mockImplementation(async (path: string) => {
    if (path === "/health") return { model_ready: true };
    if (path === "/sessions")
      return [
        {
          id: "session1",
          name: "Walking capture",
          source: "capture",
          frame_count: 2,
          revision: 7,
        },
      ];
    if (path === "/sessions/session1") return capture();
    if (path === "/sessions/session1/job") return latest;
    if (path.startsWith("/jobs/job1/resume")) return job("queued");
    if (path === "/jobs/job1") return job("running");
    throw new Error(`Unexpected API path ${path}`);
  });
}
beforeEach(() => {
  mock.api.mockReset();
  localStorage.clear();
  HTMLDialogElement.prototype.showModal = function () {
    this.setAttribute("open", "");
  };
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("saved processing workflows", () => {
  it("discovers interrupted processing and resumes remaining frames with the saved revision", async () => {
    apiDefaults(job());
    render(<App />);
    const resume = await screen.findByRole("button", {
      name: "Resume remaining frames",
    });
    expect(
      screen.getByText(/1 frames saved · 1 frames remaining/),
    ).toBeTruthy();
    fireEvent.click(resume);
    await waitFor(() =>
      expect(mock.api).toHaveBeenCalledWith(
        "/jobs/job1/resume?revision=7",
        expect.objectContaining({ method: "POST" }),
      ),
    );
    expect(
      screen.queryByRole("button", { name: "Resume remaining frames" }),
    ).toBeNull();
  });
  it("offers an in-page retry after initial connection failure", async () => {
    mock.api.mockRejectedValue(new Error("Server unavailable"));
    render(<App />);
    const retry = await screen.findByRole("button", {
      name: "Retry connection",
    });
    apiDefaults();
    fireEvent.click(retry);
    await screen.findByText("Walking capture");
    expect(
      screen.queryByRole("button", { name: "Retry connection" }),
    ).toBeNull();
  });
  it("restores the last reviewed frame on reload", async () => {
    localStorage.setItem(
      "mocap.workspace",
      JSON.stringify({ id: "session1", frameId: 1 }),
    );
    apiDefaults();
    render(<App />);
    await screen.findByText("Walking capture");
    expect(
      (
        screen.getByRole("slider", {
          name: "Current frame",
        }) as HTMLInputElement
      ).value,
    ).toBe("1");
  });
  it("loads archive list and restores a session through a revision-checked action", async () => {
    apiDefaults();
    const base = mock.api.getMockImplementation()!;
    mock.api.mockImplementation(async (path: string, init?: RequestInit) => {
      if (path === "/sessions?archived=true")
        return [
          {
            id: "session2",
            name: "Archived run",
            source: "capture",
            frame_count: 2,
            revision: 4,
            archived: true,
          },
        ];
      if (path === "/sessions/session2")
        return { ...capture(), id: "session2", revision: 4 };
      if (path === "/sessions/session2/unarchive?revision=4")
        return { ...capture(), id: "session2", revision: 5 };
      return base(path, init);
    });
    render(<App />);
    await screen.findByText("Walking capture");
    await waitFor(() =>
      expect(
        (
          screen.getByRole("button", {
            name: "Saved sessions",
          }) as HTMLButtonElement
        ).disabled,
      ).toBe(false),
    );
    fireEvent.click(screen.getByRole("button", { name: "Saved sessions" }));
    fireEvent.click(
      screen.getByRole("checkbox", { name: "Show archived sessions" }),
    );
    fireEvent.click(
      await screen.findByRole("button", { name: "Restore to workspace" }),
    );
    await waitFor(() =>
      expect(mock.api).toHaveBeenCalledWith(
        "/sessions/session2/unarchive?revision=4",
        expect.objectContaining({ method: "POST" }),
      ),
    );
  });
});

describe("joint correction safety", () => {
  function editor(frameIndex = 0, disabled = false) {
    const session = capture(),
      onEdit = vi.fn(),
      onInteraction = vi.fn();
    const props = {
      name: "cam1",
      available: ["cam1", "cam2"],
      onCamera: vi.fn(),
      view: session.frames[frameIndex].views.cam1,
      frame: session.frames[frameIndex],
      session,
      selected: 10,
      onSelect: vi.fn(),
      onEdit,
      onActor: vi.fn(),
      onInteraction,
      disabled,
      showRaw: false,
      showProjection: true,
      place: false,
    };
    return { props, onEdit, onInteraction };
  }
  it("supports keyboard nudging with explicit frame identity and clipped image bounds", () => {
    const { props, onEdit } = editor(1);
    render(<CameraView {...props} />);
    const wrist = screen.getByRole("button", { name: /Right wrist in cam1/ });
    fireEvent.keyDown(wrist, { key: "ArrowRight", shiftKey: true });
    expect(onEdit).toHaveBeenCalledWith(1, "cam1", 10, [110, 200]);
  });
  it("does not permit keyboard or numeric edits while processing", () => {
    const { props, onEdit } = editor(0, true);
    render(<CameraView {...props} />);
    fireEvent.keyDown(
      screen.getByRole("button", { name: /Right wrist in cam1/ }),
      { key: "ArrowRight" },
    );
    expect(onEdit).not.toHaveBeenCalled();
    expect(
      (
        screen.getByRole("button", {
          name: "Apply joint position in cam1",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
  });
  it("clears an unsaved numeric draft when the displayed frame changes", () => {
    const { props, onEdit } = editor();
    const { rerender } = render(<CameraView {...props} />);
    fireEvent.change(
      screen.getByRole("spinbutton", { name: "Joint X in cam1" }),
      { target: { value: "330" } },
    );
    rerender(
      <CameraView
        {...props}
        frame={props.session.frames[1]}
        view={props.session.frames[1].views.cam1}
      />,
    );
    expect(
      (
        screen.getByRole("spinbutton", {
          name: "Joint X in cam1",
        }) as HTMLInputElement
      ).value,
    ).toBe("100.0");
    fireEvent.click(
      screen.getByRole("button", { name: "Apply joint position in cam1" }),
    );
    expect(onEdit).toHaveBeenCalledWith(1, "cam1", 10, [100, 200]);
  });
  it("cancels an in-progress drag if its frame changes", () => {
    const { props, onEdit } = editor();
    const { rerender } = render(<CameraView {...props} />);
    const svg = screen.getByLabelText(
      "cam1 pose editor",
    ) as unknown as SVGSVGElement;
    svg.createSVGPoint = () =>
      ({
        x: 0,
        y: 0,
        matrixTransform() {
          return { x: 100, y: 200 };
        },
      }) as unknown as DOMPoint;
    svg.getScreenCTM = () =>
      ({
        inverse() {
          return {};
        },
      }) as DOMMatrix;
    svg.setPointerCapture = vi.fn();
    fireEvent.pointerDown(
      screen.getByRole("button", { name: /Right wrist in cam1/ }),
      { clientX: 100, clientY: 200, pointerId: 1 },
    );
    rerender(
      <CameraView
        {...props}
        frame={props.session.frames[1]}
        view={props.session.frames[1].views.cam1}
      />,
    );
    fireEvent.pointerUp(svg, { clientX: 300, clientY: 300, pointerId: 1 });
    expect(onEdit).not.toHaveBeenCalled();
  });
});
