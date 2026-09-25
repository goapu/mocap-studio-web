"""Render a side-by-side demo video from a saved result.

Left: every camera with the 3D result reprojected. Right: the metric 3D
skeleton on a 0.5 m floor grid with the camera positions, orbiting slowly.

    python scripts/render_3d_demo.py <saved_folder> demo.mp4 \
        --video cam1=cam01.mp4 --video cam2=cam02.mp4 ... --slowmo 0.5 --loops 3
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

COL = {"L": (255, 181, 63), "R": (67, 159, 255), "C": (235, 235, 235)}
BG = (26, 20, 16)


def arr(rows):
    return np.array([[np.nan if v is None else v for v in r] for r in rows], float)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("result", type=Path, help="folder saved by the app (pose3d.json)")
    p.add_argument("output", type=Path)
    p.add_argument("--video", action="append", required=True, help="camN=path")
    p.add_argument(
        "--title",
        default="Mocap Studio Live · multi-camera capture → metric 3D skeleton",
    )
    p.add_argument("--credit", default="")
    p.add_argument("--slowmo", type=float, default=0.5)
    p.add_argument("--loops", type=int, default=2)
    a = p.parse_args()

    data = json.loads((a.result / "pose3d.json").read_text())
    joints, bones = data["skeleton"]["joints"], data["skeleton"]["bones"]
    frames = data["frames"]
    n = len(frames)
    X = np.stack([arr(f["joints3d"]) for f in frames])
    sig = np.stack(
        [
            np.array([np.nan if v is None else v for v in f["sigma3d_mm"]])
            for f in frames
        ]
    )
    names = data["cameras"]
    rp = {
        c: np.stack([arr(f["views"][c]["reprojected2d"]) for f in frames])
        for c in names
    }
    videos = dict(v.split("=", 1) for v in a.video)
    caps = {c: cv2.VideoCapture(videos[c]) for c in names}
    cam_pos = []
    for c in names:
        cam = data["calibration"]["cameras"][c]
        R, T = np.array(cam["R"]), np.array(cam["T"])
        cam_pos.append((-R.T @ T, R[2]))
    z_up = data["world_frame"] == "z_up"
    to_z = (
        (lambda P: P)
        if z_up
        else (lambda P: np.stack([P[..., 0], P[..., 2], -P[..., 1]], -1))
    )
    Xz = to_z(X)
    cam_pos = [(to_z(c), to_z(f)) for c, f in cam_pos]
    floor = np.nanpercentile(Xz[..., 2], 1) if not z_up else 0.0
    center = np.nanmean(Xz[:, [11, 12]], axis=(0, 1))
    center[2] = floor + 0.85

    W, H, TOP, BOT = 1600, 900, 64, 44
    cols = 2 if len(names) <= 4 else 3
    rows = math.ceil(len(names) / cols)
    first = [caps[c].read()[1] for c in names]
    for c in names:
        caps[c].set(cv2.CAP_PROP_POS_FRAMES, 0)
    aspect = first[0].shape[1] / first[0].shape[0]
    tile_h = (H - TOP - BOT) // rows
    tile_w = int(tile_h * aspect)
    if tile_w * cols > 0.45 * W:
        tile_w = int(0.45 * W / cols)
        tile_h = int(tile_w / aspect)
    left_w = tile_w * cols + 20
    PW, PH = W - left_w - 20, H - TOP - BOT
    fv = PH * 1.05

    def view(az, el=0.22, dist=2.9):
        eye = center + dist * np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
        )
        z = center - eye
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1])
        x /= np.linalg.norm(x)
        Rv = np.stack([x, np.cross(z, x), z])
        return Rv, -Rv @ eye

    def proj(P, Rv, Tv):
        c = (Rv @ np.asarray(P).T).T + Tv
        return np.stack(
            [fv * c[:, 0] / c[:, 2] + PW / 2, fv * c[:, 1] / c[:, 2] + PH / 2], -1
        ), c[:, 2]

    def side(j):
        return (
            "L"
            if joints[j].startswith("left")
            else "R"
            if joints[j].startswith("right")
            else "C"
        )

    def pt(q):
        return tuple(int(v) for v in np.round(q))

    def render3d(i, az):
        img = np.zeros((PH, PW, 3), np.uint8)
        img[:] = (22, 17, 13)
        Rv, Tv = view(az)
        for k in np.arange(-2.5, 2.51, 0.5):
            for s, e in (([k, -2.5], [k, 2.5]), ([-2.5, k], [2.5, k])):
                seg = np.linspace(s, e, 21)
                seg = np.column_stack([seg + center[:2], np.full(21, floor)])
                q, d = proj(seg, Rv, Tv)
                for m in range(20):
                    if d[m] > 0.3 and d[m + 1] > 0.3:
                        cv2.line(
                            img,
                            pt(q[m]),
                            pt(q[m + 1]),
                            (70, 60, 52) if k % 1 == 0 else (52, 44, 38),
                            1,
                            cv2.LINE_AA,
                        )
        for cpos, fwd in cam_pos:
            q, d = proj(np.stack([cpos, cpos + 0.35 * fwd]), Rv, Tv)
            if (d > 0.3).all():
                cv2.circle(img, pt(q[0]), 6, (125, 107, 91), -1, cv2.LINE_AA)
                cv2.line(img, pt(q[0]), pt(q[1]), (125, 107, 91), 2, cv2.LINE_AA)
        P = Xz[i]
        shadow = P.copy()
        shadow[:, 2] = floor
        q, _ = proj(shadow, Rv, Tv)
        for a_, b_ in bones:
            if np.isfinite(q[[a_, b_]]).all():
                cv2.line(img, pt(q[a_]), pt(q[b_]), (40, 33, 28), 6, cv2.LINE_AA)
        q, d = proj(P, Rv, Tv)
        for k in sorted(
            range(len(bones)), key=lambda k: -np.nanmean(d[list(bones[k])])
        ):
            a_, b_ = bones[k]
            if np.isfinite(q[[a_, b_]]).all():
                cv2.line(img, pt(q[a_]), pt(q[b_]), COL[side(b_)], 7, cv2.LINE_AA)
        for j in np.argsort(-np.nan_to_num(d)):
            if np.isfinite(q[j]).all():
                s = sig[i, j]
                c = (
                    (153, 211, 52)
                    if s < 15
                    else (36, 191, 251)
                    if s < 40
                    else (113, 113, 248)
                )
                cv2.circle(img, pt(q[j]), 7, c, -1, cv2.LINE_AA)
                cv2.circle(img, pt(q[j]), 7, (20, 20, 20), 1, cv2.LINE_AA)
        return img

    cache = {c: [] for c in names}
    for c in names:
        for _ in range(n):
            ok, im = caps[c].read()
            if not ok:
                break
            r = rp[c][len(cache[c])]
            th = max(3, int(im.shape[0] / 200))
            for a_, b_ in bones:
                if np.isfinite(r[[a_, b_]]).all():
                    cv2.line(im, pt(r[a_]), pt(r[b_]), COL[side(b_)], th, cv2.LINE_AA)
            for j in range(len(r)):
                if np.isfinite(r[j]).all():
                    cv2.circle(im, pt(r[j]), th + 1, (255, 255, 255), -1, cv2.LINE_AA)
            cache[c].append(
                cv2.resize(im, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            )

    q = data["quality"]
    fps_out = min(30.0, data["fps"] * a.slowmo)
    step = data["fps"] * a.slowmo / fps_out
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is required to write the demo video.")
    proc = subprocess.Popen(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{W}x{H}",
            "-r",
            f"{fps_out:.3f}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "22",
            "-preset",
            "slow",
            "-movflags",
            "+faststart",
            str(a.output),
        ],
        stdin=subprocess.PIPE,
    )
    idxs = np.arange(0, n, step).astype(int)
    total = len(idxs) * a.loops
    font = cv2.FONT_HERSHEY_SIMPLEX
    for s_i in range(total):
        i = idxs[s_i % len(idxs)]
        az = math.radians(-110) + math.radians(100) * s_i / total
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:] = BG
        for c, name in enumerate(names):
            if i < len(cache[name]):
                x, y = 20 + (c % cols) * tile_w, TOP + (c // cols) * tile_h
                canvas[y : y + tile_h, x : x + tile_w] = cache[name][i]
        canvas[TOP : TOP + PH, left_w : left_w + PW] = render3d(i, az)
        cv2.putText(
            canvas, a.title, (20, 38), font, 0.85, (241, 235, 230), 2, cv2.LINE_AA
        )
        foot = (
            f"frame {i + 1}/{n} @ {data['fps']:.0f} fps · {a.slowmo:g}x speed · "
            f"{q['coverage_smoothed_pct']:.0f}% joints · median reprojection "
            f"{q['median_reprojection_px_smoothed']} px"
        )
        cv2.putText(
            canvas,
            foot.replace("·", "|"),
            (20, H - 16),
            font,
            0.5,
            (166, 151, 139),
            1,
            cv2.LINE_AA,
        )
        if a.credit:
            cv2.putText(
                canvas,
                a.credit,
                (left_w + 10, TOP + 24),
                font,
                0.5,
                (132, 120, 110),
                1,
                cv2.LINE_AA,
            )
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"Wrote {a.output} ({total / fps_out:.1f} s)")


if __name__ == "__main__":
    main()
