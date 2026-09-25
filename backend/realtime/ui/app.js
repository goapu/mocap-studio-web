import { StreamPlayer } from "./player.js";
import { Viewer3D } from "./viewer3d.js";

const $ = (id) => document.getElementById(id);
const api = async (path, options = {}) => {
  const res = await fetch(path, {
    headers: options.body ? { "Content-Type": "application/json" } : {},
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
};

const state = {
  system: null,
  calibration: null, // {path, cameras, world_frame}
  videos: {}, // cam -> {path, info}
  run: null,
  status: null,
  cams: [],
  lastHeader: null,
  mode: "live",
};
const LEFT = "#3fb5ff", RIGHT = "#ff9f43", CENTER = "#d9e2ec";
const viewer = new Viewer3D($("view3d"));

// ---------------------------------------------------------------- desktop bridge
const desktop = () => window.pywebview && window.pywebview.api;
const whenDesktop = new Promise((resolve) => {
  if (window.pywebview) resolve(true);
  window.addEventListener("pywebviewready", () => resolve(true));
  setTimeout(() => resolve(false), 1500);
});

async function pickFile(kind) {
  await whenDesktop;
  if (desktop()) return desktop().pick_file(kind);
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = kind === "calibration" ? ".json,.npz" : "video/*,.mp4,.mov,.avi,.mkv,.m4v";
    input.onchange = async () => {
      const file = input.files[0];
      if (!file) return resolve(null);
      try {
        resolve(await upload(file, kind));
      } catch (err) {
        showError(err.message);
        resolve(null);
      }
    };
    input.click();
  });
}

function upload(file, kind, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "api/upload");
    xhr.setRequestHeader("X-Filename", file.name);
    xhr.upload.onprogress = (e) => {
      const bar = document.querySelector(`[data-upload="${kind}"] div`);
      if (bar && e.lengthComputable) bar.style.width = `${(e.loaded / e.total) * 100}%`;
      if (onProgress) onProgress(e);
    };
    xhr.onload = () => {
      if (xhr.status >= 300) return reject(new Error("Upload failed."));
      resolve(JSON.parse(xhr.responseText).path);
    };
    xhr.onerror = () => reject(new Error("Upload failed."));
    xhr.send(file);
  });
}

// ---------------------------------------------------------------- system
async function loadSystem() {
  const sys = await api("api/system");
  state.system = sys;
  const chip = $("chip-accel");
  const coreml = sys.providers.includes("CoreMLExecutionProvider");
  const cuda = sys.providers.includes("CUDAExecutionProvider");
  chip.textContent = coreml ? "GPU · Apple CoreML" : cuda ? "GPU · CUDA" : "CPU inference";
  chip.className = `chip ${sys.gpu ? "gpu" : "cpu"}`;
  chip.title = `${sys.platform} · ${sys.providers.join(", ")}`;
  const select = $("preset");
  const current = select.value;
  select.innerHTML = "";
  for (const [key, p] of Object.entries(sys.presets)) {
    const opt = document.createElement("option");
    opt.value = key;
    const installed = sys.models[p.pose].installed && sys.models[p.detector].installed;
    opt.textContent = p.label + (installed ? "" : "  (download)");
    select.appendChild(opt);
  }
  select.value = current || "balanced";
  if (!$("save-folder").value) $("save-folder").value = sys.export_dir;
  renderModelStatus();
}

function renderModelStatus() {
  const sys = state.system;
  const preset = sys.presets[$("preset").value];
  const box = $("model-status");
  const missing = [preset.detector, preset.pose].filter((k) => !sys.models[k].installed);
  box.innerHTML = "";
  if (!missing.length) {
    box.textContent = "✓ Model installed";
  } else {
    const dl = missing.map((k) => sys.models[k].download).find((d) => d);
    if (dl && dl.state === "downloading") {
      const pct = dl.total ? Math.round((dl.bytes / dl.total) * 100) : null;
      box.textContent = `Downloading… ${pct != null ? pct + "%" : Math.round(dl.bytes / 1e6) + " MB"}`;
    } else {
      const size = missing.reduce((a, k) => a + sys.models[k].approx_mb, 0);
      const btn = document.createElement("button");
      btn.className = "btn";
      btn.textContent = `Download (~${size} MB)`;
      btn.onclick = async () => {
        for (const k of missing) await api(`api/models/${k}/download`, { method: "POST" });
        pollModels();
      };
      box.append(btn);
      if (dl && dl.state === "error") box.append(` ${dl.message}`);
    }
  }
  updateStart();
}

async function pollModels() {
  await loadSystem();
  const preset = state.system.presets[$("preset").value];
  const busy = [preset.detector, preset.pose].some(
    (k) => (state.system.models[k].download || {}).state === "downloading",
  );
  if (busy) setTimeout(pollModels, 800);
}

// ---------------------------------------------------------------- calibration & cameras
$("pick-cal").onclick = async () => {
  const path = await pickFile("calibration");
  if (!path) return;
  try {
    const cal = await api("api/calibration", { method: "POST", body: { path } });
    state.calibration = { path, ...cal };
    state.videos = {};
    $("cal-name").textContent = path.split(/[\\/]/).pop();
    const q = cal.quality || {};
    $("cal-info").innerHTML =
      `${Object.keys(cal.cameras).length} cameras · world frame <b>${cal.world_frame}</b>` +
      (q.stereo_rms_px != null ? ` · stereo RMS ${q.stereo_rms_px.toFixed(2)} px` : "") +
      (cal.world_frame === "camera"
        ? `<div class="warn">No floor in this calibration: the 3D grid is estimated from the feet.</div>`
        : "");
    renderCameraRows();
    showError("");
  } catch (err) {
    showError(err.message);
  }
};

function renderCameraRows() {
  const list = $("camera-list");
  list.innerHTML = "";
  for (const [name, cam] of Object.entries(state.calibration.cameras)) {
    const row = document.createElement("div");
    row.className = "cam-row";
    const v = state.videos[name];
    const info = v && v.info;
    let meta = `<span>Calibrated ${cam.image_size[0]}×${cam.image_size[1]}</span>`;
    let metaClass = "meta";
    if (info && info.error) {
      meta = info.error;
      metaClass = "meta err";
    } else if (info) {
      const aspectOk =
        Math.abs(info.width / info.height - cam.image_size[0] / cam.image_size[1]) < 0.01;
      meta =
        `<span>${info.width}×${info.height} · ${info.fps.toFixed(2)} fps · ${info.frames} frames</span>` +
        (aspectOk ? "" : "<span>aspect ≠ calibration</span>");
      if (!aspectOk) metaClass = "meta err";
    }
    row.innerHTML = `
      <span class="name">${name}</span>
      <div class="filepick"><button class="btn">${v ? "Change…" : "Choose video…"}</button>
        <span class="file">${v ? v.path.split(/[\\/]/).pop() : "—"}</span></div>
      <div class="${metaClass}">${meta}
        <label class="offset" title="Skip this many frames at the start to align cameras">sync +<input type="number" min="0" step="1" value="${v ? v.offset || 0 : 0}" data-offset="${name}"/></label></div>
      <div class="upload-bar" data-upload="${name}"><div></div></div>`;
    row.querySelector("button").onclick = async () => {
      const path = await pickFile(name);
      if (!path) return;
      const probe = await api("api/probe", { method: "POST", body: { videos: { [name]: path } } });
      state.videos[name] = { path, info: probe[name], offset: 0 };
      renderCameraRows();
      checkFps();
    };
    row.querySelector("[data-offset]").onchange = (e) => {
      if (state.videos[name]) state.videos[name].offset = Math.max(0, parseInt(e.target.value || 0));
    };
    list.appendChild(row);
  }
  updateStart();
}

function checkFps() {
  const fps = Object.values(state.videos).map((v) => v.info && v.info.fps).filter(Boolean);
  if (fps.length > 1 && Math.max(...fps) - Math.min(...fps) > 0.01 * Math.max(...fps)) {
    showError("Camera frame rates differ. Record all cameras at the same constant frame rate.");
  } else showError("");
}

function updateStart() {
  const ready = Object.values(state.videos).filter((v) => v.info && !v.info.error).length >= 2;
  const sys = state.system;
  const preset = sys && sys.presets[$("preset").value];
  const installed = preset && sys.models[preset.pose].installed && sys.models[preset.detector].installed;
  const running = state.status && ["starting", "running", "finalizing"].includes(state.status.status);
  $("start").disabled = !(ready && installed && state.calibration) || running;
  $("start").title = !installed ? "Download the selected model first" : ready ? "" : "Choose at least two camera videos";
}

$("preset").onchange = renderModelStatus;
$("smoothing").oninput = () => {
  $("smooth-out").textContent = `${(2 ** $("smoothing").value).toFixed(2)}×`;
};

// ---------------------------------------------------------------- run
$("start").onclick = async () => {
  showError("");
  const videos = {}, offsets = {};
  for (const [name, v] of Object.entries(state.videos)) {
    if (v.info && !v.info.error) {
      videos[name] = v.path;
      offsets[name] = v.offset || 0;
    }
  }
  try {
    const run = await api("api/runs", {
      method: "POST",
      body: {
        calibration_path: state.calibration.path,
        videos,
        options: {
          name: $("run-name").value || "capture",
          preset: $("preset").value,
          flip_test: $("flip").checked,
          // slider left = smoother (lower process noise)
          smoothing: 2 ** Number($("smoothing").value),
          min_score: Number($("min-score").value),
          det_interval: Number($("det-interval").value),
          preview_width: Number($("preview-width").value),
          live_bone_constraint: $("live-bones").checked,
          realtime_pace: $("pace").checked,
          offsets,
        },
      },
    });
    state.run = run;
    state.status = run;
    state.rigSet = false;
    state.skeleton = null;
    $("welcome").classList.add("hidden");
    $("save-status").textContent = "";
    setupCams(Object.keys(videos));
    startStream(`api/runs/${run.id}/stream`, "live");
    pollStatus();
  } catch (err) {
    showError(err.message);
  }
};

$("stop").onclick = () => state.run && api(`api/runs/${state.run.id}/stop`, { method: "POST" });

$("replay").onclick = () => {
  if (!state.run) return;
  startStream(`api/runs/${state.run.id}/replay?variant=smoothed`, "smoothed");
};

const player = new StreamPlayer({
  onFrame: (header, bitmaps) => drawFrame(header, bitmaps),
  onPacket: (header) => {
    if (header.stats) updateMetrics(header.stats);
    const total = header.total || 1;
    if (state.mode === "live") $("bar-processed").style.width = `${((header.frame + 1) / total) * 100}%`;
  },
  onEnd: () => {
    $("play").disabled = true;
  },
});

function startStream(url, mode) {
  state.mode = mode;
  $("play").disabled = false;
  $("play").textContent = "Pause";
  player.mode = $("mode").value;
  player.open(url).catch((err) => showError(err.message));
}

$("play").onclick = () => {
  player.playing = !player.playing;
  $("play").textContent = player.playing ? "Pause" : "Play";
};
$("mode").onchange = () => (player.mode = $("mode").value);
$("follow").onchange = () => (viewer.follow = $("follow").checked);
$("reset-view").onclick = () => viewer.resetView();

async function pollStatus() {
  if (!state.run) return;
  try {
    const s = await api(`api/runs/${state.run.id}`);
    state.status = s;
    const chip = $("chip-run");
    chip.textContent = { starting: "Loading models", running: "Processing", finalizing: "Smoothing", finished: "Finished", stopped: "Stopped", error: "Error" }[s.status] || s.status;
    chip.className = `chip ${s.status === "error" ? "error" : ["running", "starting", "finalizing"].includes(s.status) ? "running" : ""}`;
    if (s.skeleton && s.rig && !state.rigSet) {
      state.skeleton = s.skeleton;
      viewer.setSkeleton(s.skeleton, s.world_frame);
      viewer.setRig(s.rig);
      state.rigSet = true;
    }
    updateMetrics(s.stats);
    const running = ["starting", "running", "finalizing"].includes(s.status);
    $("start").classList.toggle("hidden", running);
    $("stop").classList.toggle("hidden", !running);
    $("sidebar").querySelectorAll(".panel:not(#save-panel) button, .panel:not(#save-panel) input, .panel:not(#save-panel) select").forEach((el) => {
      if (el.id !== "stop") el.disabled = running;
    });
    $("save").disabled = !s.has_final;
    $("replay").disabled = !s.has_final;
    if (s.status === "error") showError(s.error || "Processing failed.");
    if (s.status === "starting" || s.status === "finalizing") showInfo(s.message);
    else if (s.status !== "error") showInfo("");
    if (s.quality) {
      $("m-cov").textContent = `${s.quality.coverage_smoothed_pct}%`;
    }
    if (running) setTimeout(pollStatus, 500);
    else {
      updateStart();
      if (s.status !== "error") $("chip-run").title = s.message;
    }
  } catch (err) {
    setTimeout(pollStatus, 1500);
  }
}

// ---------------------------------------------------------------- drawing
function setupCams(names) {
  const box = $("cams");
  box.innerHTML = "";
  box.className = `cams n${names.length}`;
  state.lastFrame = null;
  state.cams = names.map((name) => {
    const div = document.createElement("div");
    div.className = "camview";
    div.innerHTML = `<canvas></canvas><span class="tag">${name}</span>`;
    box.appendChild(div);
    const canvas = div.querySelector("canvas");
    const entry = { name, canvas, ctx: canvas.getContext("2d"), tag: div.querySelector(".tag") };
    entry.placeholder = true;
    new ResizeObserver(() => {
      const r = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, canvas.clientWidth * r);
      canvas.height = Math.max(1, canvas.clientHeight * r);
      if (state.lastFrame) drawFrame(...state.lastFrame, true);
      else drawPlaceholder(entry);
    }).observe(canvas);
    return entry;
  });
}

function drawPlaceholder({ canvas, ctx }) {
  ctx.fillStyle = "#07090c";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#5f6b7a";
  ctx.font = `${13 * (window.devicePixelRatio || 1)}px -apple-system, sans-serif`;
  ctx.textAlign = "center";
  ctx.fillText("Waiting for the first processed frame…", canvas.width / 2, canvas.height / 2);
  ctx.textAlign = "start";
}

function sideColor(name) {
  return name.startsWith("left") ? LEFT : name.startsWith("right") ? RIGHT : CENTER;
}

function drawFrame(header, bitmaps, redraw = false) {
  if (!redraw && state.lastFrame) state.lastFrame[1].forEach((b) => b.close && b.close());
  state.lastFrame = [header, bitmaps];
  const sk = state.skeleton;
  const minScore = Number($("min-score").value);
  header.cams.forEach((cam, c) => {
    const view = state.cams[c];
    const bmp = bitmaps[c];
    if (!view || !bmp) return;
    const { canvas, ctx } = view;
    const W = canvas.width, H = canvas.height;
    const s = Math.min(W / cam.w, H / cam.h);
    const ox = (W - cam.w * s) / 2, oy = (H - cam.h * s) / 2;
    ctx.fillStyle = "#07090c";
    ctx.fillRect(0, 0, W, H);
    ctx.drawImage(bmp, ox, oy, cam.w * s, cam.h * s);
    const P = (p) => [ox + p[0] * s, oy + p[1] * s];
    const r = window.devicePixelRatio || 1;
    if (!sk) return;
    if ($("show-reproj").checked && cam.reproj) {
      ctx.lineWidth = 2.2 * r;
      for (const [a, b] of sk.bones) {
        const pa = cam.reproj[a], pb = cam.reproj[b];
        if (!pa || !pb || pa[0] === null || pb[0] === null) continue;
        ctx.strokeStyle = sideColor(sk.joints[b]);
        ctx.globalAlpha = 0.9;
        ctx.beginPath();
        ctx.moveTo(...P(pa));
        ctx.lineTo(...P(pb));
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    if ($("show-2d").checked && cam.kp) {
      cam.kp.forEach((k, j) => {
        if (!k || k[0] === null || k[2] < minScore) return;
        const [x, y] = P(k);
        const used = cam.used ? cam.used[j] : true;
        ctx.beginPath();
        ctx.arc(x, y, 3.2 * r, 0, Math.PI * 2);
        ctx.fillStyle = used ? "#ffffff" : "#f87171";
        ctx.fill();
        ctx.lineWidth = 1.2 * r;
        ctx.strokeStyle = "#000";
        ctx.stroke();
      });
    }
    view.tag.textContent = `${cam.name} · ${formatTime(header.t)} · #${header.frame}`;
  });
  viewer.update(header.X, header.sigma_mm);
  $("time-now").textContent = formatTime(header.t);
  const total = header.total || 1;
  $("time-total").textContent = formatTime(total / header.fps);
  $("bar-played").style.width = `${((header.frame + 1) / total) * 100}%`;
  if (state.mode === "smoothed") $("bar-processed").style.width = "100%";
}

function formatTime(t) {
  const m = Math.floor(t / 60);
  return `${m}:${(t - m * 60).toFixed(2).padStart(5, "0")}`;
}

function updateMetrics(s) {
  if (!s) return;
  const set = (id, v, cls) => {
    const el = $(id);
    el.textContent = v;
    el.className = cls || "";
  };
  set("m-fps", s.processing_fps ? s.processing_fps.toFixed(1) : "—");
  const rt = s.realtime_factor;
  set("m-rt", rt ? `${rt.toFixed(2)}×` : "—", rt >= 1 ? "good" : rt >= 0.5 ? "warn" : rt ? "bad" : "");
  set("m-lat", s.latency_ms ? `${Math.round(s.latency_ms)} ms` : "—");
  set("m-cov", s.coverage_3d ? `${s.coverage_3d}%` : "—");
  set("m-rep", s.mean_reproj_px != null ? `${s.mean_reproj_px} px` : "—");
  set("m-lr", String(s.left_right_repairs ?? "—"));
}

function frameLoop(now) {
  player.tick(now);
  $("m-play").textContent = player.started ? `${player.rate.toFixed(2)}×` : "—";
  $("m-play").className = !player.started ? "" : player.rate > 0.97 ? "good" : player.rate > 0.5 ? "warn" : "bad";
  $("m-buf").textContent = player.started ? `${player.bufferedSeconds.toFixed(1)} s` : "—";
  requestAnimationFrame(frameLoop);
}
requestAnimationFrame(frameLoop);

// ---------------------------------------------------------------- save
$("pick-folder").onclick = async () => {
  await whenDesktop;
  if (desktop()) {
    const folder = await desktop().pick_folder();
    if (folder) $("save-folder").value = folder;
  } else {
    showSave("In the browser, type a folder path on this computer (the desktop app shows a folder picker).");
  }
};

$("save").onclick = async () => {
  if (!state.run) return;
  $("save").disabled = true;
  try {
    await api(`api/runs/${state.run.id}/save`, {
      method: "POST",
      body: {
        folder: $("save-folder").value.trim(),
        options: {
          csv: $("opt-csv").checked,
          trc: $("opt-trc").checked,
          videos: $("opt-videos").checked,
          bone_constraint: $("opt-bones").checked,
        },
      },
    });
    pollSave();
  } catch (err) {
    showSave(err.message, true);
    $("save").disabled = false;
  }
};

async function pollSave() {
  const s = await api(`api/runs/${state.run.id}/save`);
  if (s.status === "done") {
    const box = $("save-status");
    box.innerHTML = `Saved ${s.files.length} files to<br><b>${s.folder}</b><br>`;
    const link = document.createElement("a");
    link.textContent = "Open folder";
    link.onclick = () => api("api/open", { method: "POST", body: { path: s.folder } }).catch((e) => showSave(e.message, true));
    box.appendChild(link);
    $("save").disabled = false;
  } else if (s.status === "error") {
    showSave(s.message, true);
    $("save").disabled = false;
  } else {
    showSave(`Saving… ${s.progress}%`);
    setTimeout(pollSave, 400);
  }
}

function showSave(text, error) {
  $("save-status").textContent = text;
  $("save-status").style.color = error ? "var(--poor)" : "";
}
function showError(text) {
  $("setup-error").textContent = text;
  $("setup-error").classList.remove("info");
}
function showInfo(text) {
  $("setup-error").textContent = text;
  $("setup-error").classList.add("info");
}

loadSystem().catch((err) => showError(`Cannot reach the local engine: ${err.message}`));
