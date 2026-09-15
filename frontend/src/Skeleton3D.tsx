import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RotateCcw, Maximize2 } from "lucide-react";
import type { Frame, Session } from "./types";

export default function Skeleton3D({
  frame,
  session,
  selected,
  onSelect,
}: {
  frame: Frame;
  session: Session;
  selected: number;
  onSelect: (j: number) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const state = useRef<{
    scene: THREE.Scene;
    group: THREE.Group;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    render: () => void;
  } | null>(null);
  const selectedCallback = useRef(onSelect);
  selectedCallback.current = onSelect;
  const [failure, setFailure] = useState("");
  useEffect(() => {
    if (!host.current) return;
    const el = host.current;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      setFailure(
        "3D rendering requires WebGL. Your reconstruction remains available in the joint table and export.",
      );
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#182124");
    const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 1000);
    camera.position.set(3.6, 2.5, 4.7);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.9, 0);
    controls.enableDamping = false;
    controls.minDistance = 0.3;
    controls.maxDistance = 100;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x53686a, 3));
    const light = new THREE.DirectionalLight(0xffffff, 3);
    light.position.set(3, 6, 4);
    scene.add(light);
    const group = new THREE.Group();
    scene.add(group);
    const render = () => renderer.render(scene, camera);
    state.current = { scene, group, camera, controls, render };
    const resize = () => {
      const w = el.clientWidth,
        h = el.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      render();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    controls.addEventListener("change", render);
    const ray = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const click = (e: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.set(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        (-(e.clientY - rect.top) / rect.height) * 2 + 1,
      );
      ray.setFromCamera(mouse, camera);
      const hit = ray
        .intersectObjects(group.children)
        .find((h) => h.object.userData.joint !== undefined);
      if (hit) selectedCallback.current(hit.object.userData.joint);
    };
    renderer.domElement.addEventListener("click", click);
    resize();
    return () => {
      observer.disconnect();
      controls.dispose();
      scene.traverse((o) => {
        if (o instanceof THREE.Mesh || o instanceof THREE.LineSegments) {
          o.geometry.dispose();
          const m = o.material;
          if (Array.isArray(m)) m.forEach((v) => v.dispose());
          else m.dispose();
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
      state.current = null;
    };
  }, []);
  const previousSession = useRef("");
  useEffect(() => {
    const s = state.current;
    if (!s) return;
    while (s.group.children.length) {
      const child = s.group.children[0];
      s.group.remove(child);
      if (child instanceof THREE.Mesh || child instanceof THREE.LineSegments) {
        child.geometry.dispose();
        const m = child.material;
        if (Array.isArray(m)) m.forEach((v) => v.dispose());
        else m.dispose();
      }
    }
    const world = (p: number[]) =>
      session.calibration.world_frame === "z_up"
        ? new THREE.Vector3(p[0], p[2], -p[1])
        : new THREE.Vector3(p[0], -p[1], -p[2]);
    const points = frame.pose.map((j) => (j.point ? world(j.point) : null));
    const valid = points.filter((p): p is THREE.Vector3 => p !== null);
    const center = valid.length
      ? valid
          .reduce((a, b) => a.add(b), new THREE.Vector3())
          .multiplyScalar(1 / valid.length)
      : new THREE.Vector3();
    const bottom = valid.length ? Math.min(...valid.map((p) => p.y)) : 0;
    const grid = new THREE.GridHelper(6, 24, 0x53615a, 0x303f40);
    grid.position.set(
      center.x,
      session.calibration.world_frame === "z_up" ? 0 : bottom - 0.07,
      center.z,
    );
    s.group.add(grid);
    session.bones.forEach(([a, b]) => {
      const start = points[a],
        end = points[b];
      if (!start || !end) return;
      const dir = end.clone().sub(start);
      const geometry = new THREE.CylinderGeometry(
        0.012,
        0.012,
        Math.max(dir.length(), 0.001),
        8,
      );
      const color = a % 2 === 1 ? 0x5bc9df : 0xc2ee8c;
      const mesh = new THREE.Mesh(
        geometry,
        new THREE.MeshStandardMaterial({ color, roughness: 0.5 }),
      );
      mesh.position.copy(start).add(end).multiplyScalar(0.5);
      mesh.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        dir.normalize(),
      );
      s.group.add(mesh);
    });
    if (points[0] && points[5] && points[6]) {
      const neck = points[5].clone().add(points[6]).multiplyScalar(0.5);
      const dir = points[0].clone().sub(neck);
      const mesh = new THREE.Mesh(
        new THREE.CylinderGeometry(0.012, 0.012, dir.length(), 8),
        new THREE.MeshStandardMaterial({ color: 0xc2ee8c }),
      );
      mesh.position.copy(neck).add(points[0]).multiplyScalar(0.5);
      mesh.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        dir.normalize(),
      );
      s.group.add(mesh);
    }
    points.forEach((point, i) => {
      if (!point) return;
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(i === selected ? 0.04 : 0.022, 16, 12),
        new THREE.MeshStandardMaterial({
          color:
            i === selected
              ? 0xffffff
              : frame.pose[i].status === "corrected"
                ? 0xffc675
                : 0xc2ee8c,
        }),
      );
      mesh.position.copy(point);
      mesh.userData.joint = i;
      s.group.add(mesh);
    });
    if (previousSession.current !== session.id && valid.length) {
      s.controls.target.copy(center);
      s.camera.position.copy(center).add(new THREE.Vector3(2.5, 1.5, 3.3));
      s.controls.update();
      previousSession.current = session.id;
    }
    s.render();
  }, [frame, session, selected]);
  function reset() {
    const s = state.current;
    if (s) {
      s.camera.position
        .copy(s.controls.target)
        .add(new THREE.Vector3(2.5, 1.5, 3.3));
      s.controls.update();
      s.render();
    }
  }
  return (
    <div className="three-wrap">
      <div ref={host} className="three-host" />
      {failure && <div className="webgl-error">{failure}</div>}
      <div className="three-tools">
        <button title="Reset view" aria-label="Reset 3D view" onClick={reset}>
          <RotateCcw size={15} />
        </button>
        <button
          title="Full screen"
          aria-label="Full screen skeleton"
          onClick={() => host.current?.requestFullscreen()}
        >
          <Maximize2 size={15} />
        </button>
      </div>
      <div className="axis-key">
        <span>X</span>
        <span>Y</span>
        <span>Z</span>
        <small>
          Meters ·{" "}
          {session.calibration.world_frame === "z_up"
            ? "Z-up world"
            : "Camera reference"}
        </small>
      </div>
      <div className="three-hint">Drag to orbit · Scroll to zoom</div>
    </div>
  );
}
