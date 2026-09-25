import * as THREE from "./vendor/three.module.min.js";

const COLORS = {
  left: new THREE.Color("#3fb5ff"),
  right: new THREE.Color("#ff9f43"),
  center: new THREE.Color("#d9e2ec"),
  good: new THREE.Color("#34d399"),
  ok: new THREE.Color("#fbbf24"),
  poor: new THREE.Color("#f87171"),
};

// World (calibration) coordinates -> three.js (Y up).
function toThree(p, frame) {
  if (frame === "z_up") return new THREE.Vector3(p[0], p[2], -p[1]);
  return new THREE.Vector3(p[0], -p[1], -p[2]);
}

export class Viewer3D {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 200);
    this.frame = "camera";
    this.target = new THREE.Vector3(0, 1, 0);
    this.orbit = { theta: 0.7, phi: 1.15, radius: 5.5 };
    this.follow = true;
    this.floorY = null;
    this.ankleMins = [];
    this.trail = [];

    this.grid = new THREE.GridHelper(20, 40, 0x2c3947, 0x1a222c);
    this.scene.add(this.grid);
    this.floor = new THREE.Mesh(
      new THREE.CircleGeometry(10, 64),
      new THREE.MeshBasicMaterial({ color: 0x0c1016, transparent: true, opacity: 0.6 }),
    );
    this.floor.rotation.x = -Math.PI / 2;
    this.floor.position.y = -0.001;
    this.scene.add(this.floor);

    this.jointMesh = null;
    this.boneMesh = null;
    this.rigGroup = new THREE.Group();
    this.scene.add(this.rigGroup);
    const trailGeom = new THREE.BufferGeometry();
    trailGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(180 * 3), 3));
    this.trailLine = new THREE.Line(
      trailGeom,
      new THREE.LineBasicMaterial({ color: 0x3fb5ff, transparent: true, opacity: 0.35 }),
    );
    this.scene.add(this.trailLine);

    this._bindControls();
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
    const loop = () => {
      this.render();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  setSkeleton(skeleton, worldFrame) {
    this.frame = worldFrame || "camera";
    this.skeleton = skeleton;
    for (const m of [this.jointMesh, this.boneMesh]) if (m) this.scene.remove(m);
    const J = skeleton.joints.length;
    this.jointMesh = new THREE.InstancedMesh(
      new THREE.SphereGeometry(0.028, 16, 12),
      new THREE.MeshBasicMaterial(),
      J,
    );
    this.boneMesh = new THREE.InstancedMesh(
      new THREE.CylinderGeometry(0.014, 0.014, 1, 10, 1, true),
      new THREE.MeshBasicMaterial(),
      skeleton.bones.length,
    );
    for (let b = 0; b < skeleton.bones.length; b++) {
      const name = skeleton.joints[skeleton.bones[b][1]];
      const side = name.startsWith("left") ? "left" : name.startsWith("right") ? "right" : "center";
      this.boneMesh.setColorAt(b, COLORS[side]);
    }
    this.scene.add(this.jointMesh, this.boneMesh);
    this.trail = [];
    this.floorY = this.frame === "z_up" ? 0 : null;
    this.ankleMins = [];
    this.hide();
  }

  setRig(rig) {
    this.rigGroup.clear();
    if (!rig) return;
    const mat = new THREE.LineBasicMaterial({ color: 0x5b6b7d });
    for (const cam of rig) {
      if (!cam.center || cam.center.includes(null)) continue;
      const c = toThree(cam.center, this.frame);
      const f = toThree(cam.forward, this.frame).normalize();
      const up = toThree(cam.up, this.frame).normalize();
      const right = new THREE.Vector3().crossVectors(f, up).normalize();
      const d = 0.35, w = 0.22, h = 0.14;
      const corners = [
        [1, 1], [1, -1], [-1, -1], [-1, 1],
      ].map(([a, b]) => c.clone().add(f.clone().multiplyScalar(d)).add(right.clone().multiplyScalar(a * w)).add(up.clone().multiplyScalar(b * h)));
      const pts = [];
      for (let i = 0; i < 4; i++) pts.push(c, corners[i], corners[i], corners[(i + 1) % 4]);
      const g = new THREE.BufferGeometry().setFromPoints(pts);
      this.rigGroup.add(new THREE.LineSegments(g, mat));
    }
  }

  hide() {
    if (!this.jointMesh) return;
    const zero = new THREE.Matrix4().makeScale(0, 0, 0);
    for (let i = 0; i < this.jointMesh.count; i++) this.jointMesh.setMatrixAt(i, zero);
    for (let i = 0; i < this.boneMesh.count; i++) this.boneMesh.setMatrixAt(i, zero);
    this.jointMesh.instanceMatrix.needsUpdate = true;
    this.boneMesh.instanceMatrix.needsUpdate = true;
  }

  update(X, sigmaMm) {
    if (!this.jointMesh || !X) return;
    const pts = X.map((p) => (p && p[0] !== null ? toThree(p, this.frame) : null));
    const m = new THREE.Matrix4();
    const zero = new THREE.Matrix4().makeScale(0, 0, 0);
    pts.forEach((p, i) => {
      if (!p) return this.jointMesh.setMatrixAt(i, zero);
      m.makeTranslation(p.x, p.y, p.z);
      this.jointMesh.setMatrixAt(i, m);
      const s = sigmaMm ? sigmaMm[i] : null;
      const color = s == null ? COLORS.poor : s < 15 ? COLORS.good : s < 40 ? COLORS.ok : COLORS.poor;
      this.jointMesh.setColorAt(i, color);
    });
    const up = new THREE.Vector3(0, 1, 0);
    const q = new THREE.Quaternion();
    this.skeleton.bones.forEach(([a, b], k) => {
      const pa = pts[a], pb = pts[b];
      if (!pa || !pb) return this.boneMesh.setMatrixAt(k, zero);
      const dir = pb.clone().sub(pa);
      const len = dir.length();
      q.setFromUnitVectors(up, dir.normalize());
      m.compose(pa.clone().add(pb).multiplyScalar(0.5), q, new THREE.Vector3(1, len, 1));
      this.boneMesh.setMatrixAt(k, m);
    });
    this.jointMesh.instanceMatrix.needsUpdate = true;
    if (this.jointMesh.instanceColor) this.jointMesh.instanceColor.needsUpdate = true;
    this.boneMesh.instanceMatrix.needsUpdate = true;

    // Pelvis trail and follow target.
    const lh = this.skeleton.joints.indexOf("left_hip");
    const rh = this.skeleton.joints.indexOf("right_hip");
    if (pts[lh] && pts[rh]) {
      const pelvis = pts[lh].clone().add(pts[rh]).multiplyScalar(0.5);
      this.trail.push(pelvis);
      if (this.trail.length > 180) this.trail.shift();
      if (this.follow) this.target.lerp(pelvis, 0.08);
      const attr = this.trailLine.geometry.attributes.position;
      this.trail.forEach((p, i) => attr.setXYZ(i, p.x, (this.floorY ?? p.y) + 0.002, p.z));
      this.trailLine.geometry.setDrawRange(0, this.trail.length);
      attr.needsUpdate = true;
    }
    // Reference floor for camera-frame calibrations: lowest recent foot height.
    if (this.frame !== "z_up") {
      const feet = pts.filter((p, i) => p && /ankle|heel|toe/.test(this.skeleton.joints[i]));
      if (feet.length) {
        this.ankleMins.push(Math.min(...feet.map((p) => p.y)));
        if (this.ankleMins.length > 300) this.ankleMins.shift();
        const sorted = [...this.ankleMins].sort((a, b) => a - b);
        this.floorY = sorted[Math.floor(sorted.length * 0.05)] - 0.06;
      }
    }
    if (this.floorY != null) {
      this.grid.position.y = this.floorY;
      this.floor.position.y = this.floorY - 0.001;
    }
  }

  resetView() {
    this.orbit = { theta: 0.7, phi: 1.15, radius: 5.5 };
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  render() {
    const { theta, phi, radius } = this.orbit;
    const t = this.target;
    this.camera.position.set(
      t.x + radius * Math.sin(phi) * Math.sin(theta),
      t.y + radius * Math.cos(phi),
      t.z + radius * Math.sin(phi) * Math.cos(theta),
    );
    this.camera.lookAt(t);
    this.renderer.render(this.scene, this.camera);
  }

  _bindControls() {
    let drag = null;
    const el = this.canvas;
    el.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
      el.setPointerCapture(e.pointerId);
      el.style.cursor = "grabbing";
    });
    el.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX;
      drag.y = e.clientY;
      if (drag.pan) {
        const s = this.orbit.radius * 0.0015;
        const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0);
        const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 1);
        this.target.add(right.multiplyScalar(-dx * s)).add(up.multiplyScalar(dy * s));
      } else {
        this.orbit.theta -= dx * 0.008;
        this.orbit.phi = Math.min(Math.PI - 0.05, Math.max(0.05, this.orbit.phi - dy * 0.008));
      }
    });
    const end = () => {
      drag = null;
      el.style.cursor = "grab";
    };
    el.addEventListener("pointerup", end);
    el.addEventListener("pointercancel", end);
    el.addEventListener("contextmenu", (e) => e.preventDefault());
    el.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.orbit.radius = Math.min(40, Math.max(0.8, this.orbit.radius * Math.exp(e.deltaY * 0.001)));
      },
      { passive: false },
    );
    el.addEventListener("dblclick", () => this.resetView());
  }
}
