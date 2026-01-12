"""
Action Accuracy evaluation (ViPE + Sim(3) Umeyama + RPE).

Protocol:
1) For each episode e: (gt_video, action_seq)
2) For each model m: gen_video produced from same init frame + same action_seq
3) Reconstruct camera trajectories from pixels using ViPE:
      T_gt(t)  = ViPE(gt_video)
      T_gen(t) = ViPE(gen_video)
4) Align gen -> gt using Sim(3) Umeyama on camera centers (positions).
5) Compute Relative Pose Error (RPE) at delta frames:
      E_i = (dT_gt(i))^{-1} dT_gen_aligned(i), where dT(i) = T(i)^{-1} T(i+delta)
   Report:
      RPE-trans = || t(E_i) ||
      RPE-rot   = angle(R(E_i)) in degrees

Outputs:
- CSV table: rows for (model, bucket) where bucket is __overall__ plus optional action/group buckets.

Assumptions:
- ViPE is installed and `vipe infer` works, OR you can run it via --vipe_mode runpy with --vipe_repo.
- To parse poses robustly, we prefer COLMAP export (images.txt). If ViPE output doesn't include it directly,
  we can call scripts/vipe_to_colmap.py (requires --vipe_repo).

Manifest format (JSONL):
{"episode_id":"000001","gt_video":"/path/to/gt/000001.mp4","actions":["forward","forward","cam_left", ...]}

Generated video default pattern:
{gen_root}/{model}/{episode_id}.mp4

python eval_action_accuracy_vipe.py \
  --manifest /scratch/e0795287/AWAN/DATASET/MIND/world_model_1st/val_manifest_97_vipe.jsonl \
  --models dummy \
  --gen_root /scratch/e0795287/AWAN/Outputs/gen_videos/action_bi_mind/1st_5e-5_bz4_gpu4_full_20251127_194628/checkpoint_model_002000 \
  --gen_pattern "{gen_root}/{episode_id}_s0.mp4" \
  --cache_dir /scratch/e0795287/AWAN/Outputs/eval/cache_vipe \
  --out_csv /scratch/e0795287/AWAN/Outputs/eval/action_bi_mind/1st_5e-5_bz4_gpu4_full_20251127_194628/checkpoint_model_002000/action_accuracy.csv \
  --per_action \
  --try_vipe_to_colmap \
  --vipe_repo /hpctmp/e0795287/LWM/evalution/vipe 
"""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# Episode IO
# -----------------------------

@dataclass
class Episode:
    episode_id: str
    gt_video: Path
    actions: List[str]

def load_manifest_jsonl(p: Path) -> List[Episode]:
    out: List[Episode] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(Episode(
                episode_id=str(obj["episode_id"]),
                gt_video=Path(obj["gt_video"]),
                actions=list(obj["actions"]),
            ))
    return out

# -----------------------------
# SE(3) helpers
# -----------------------------

def quat_to_R_wxyz(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    q /= n
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)

def make_T(R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_wc
    T[:3, 3] = t_wc
    return T

def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def rot_angle_deg(R: np.ndarray) -> float:
    tr = np.trace(R)
    cos = (tr - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))

# -----------------------------
# Sim(3) Umeyama alignment
# -----------------------------

def umeyama_sim3(X: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate similarity transform (s, R, t) such that:
        Y ≈ s * R * X + t
    X, Y: (N,3)
    """
    assert X.shape == Y.shape and X.shape[1] == 3
    N = X.shape[0]
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc = X - muX
    Yc = Y - muY
    varX = (Xc * Xc).sum() / N
    Sigma = (Yc.T @ Xc) / N
    U, D, Vt = np.linalg.svd(Sigma)

    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (varX + 1e-12))
    t = muY - s * (R @ muX)
    return s, R, t

def apply_sim3_to_poses(Ts: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Apply Sim(3) to camera->world SE(3) poses:
      p' = s R p + t
      R' = R R_pose
    """
    out = Ts.copy()
    out[:, :3, :3] = R @ out[:, :3, :3]
    out[:, :3, 3] = (s * (R @ out[:, :3, 3].T)).T + t[None, :]
    return out

# -----------------------------
# RPE
# -----------------------------

def compute_rpe(gt_T: np.ndarray, est_T: np.ndarray, delta: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Relative Pose Error per step:
      dQ = Q_i^{-1} Q_{i+Δ}
      dP = P_i^{-1} P_{i+Δ}
      E  = dQ^{-1} dP
    Return arrays:
      trans_err: ||t(E)||
      rot_err_deg: angle(R(E))
    """
    n = min(len(gt_T), len(est_T))
    gt_T = gt_T[:n]
    est_T = est_T[:n]
    m = n - delta
    trans = np.zeros((m,), dtype=np.float64)
    rotdeg = np.zeros((m,), dtype=np.float64)
    for i in range(m):
        dQ = inv_T(gt_T[i]) @ gt_T[i + delta]
        dP = inv_T(est_T[i]) @ est_T[i + delta]
        E = inv_T(dQ) @ dP
        trans[i] = np.linalg.norm(E[:3, 3])
        rotdeg[i] = rot_angle_deg(E[:3, :3])
    return trans, rotdeg

# -----------------------------
# ViPE runner + pose parsing
# -----------------------------

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def ensure_vipe(video: Path, out_dir: Path, vipe_mode: str, vipe_repo: Optional[Path],
                pipeline: str, pose_only: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / "_DONE"
    if done.exists():
        return

    if vipe_mode == "cli":
        run_cmd(["vipe", "infer", str(video), "--output", str(out_dir), "--pipeline", pipeline])
    elif vipe_mode == "runpy":
        if vipe_repo is None:
            raise ValueError("--vipe_repo is required for --vipe_mode runpy")
        run_py = vipe_repo / "run.py"
        cmd = [
            "python", str(run_py),
            f"pipeline={pipeline}",
            "streams=raw_mp4_stream",
            f"streams.base_path={str(video)}",
            f"pipeline.output.output_dir={str(out_dir)}",
        ]
        if pose_only:
            cmd.append("pipeline.post.depth_align_model=null")
        run_cmd(cmd, cwd=vipe_repo)
    else:
        raise ValueError(f"Unknown vipe_mode={vipe_mode}")

    done.write_text("ok\n", encoding="utf-8")

def find_images_txt(root: Path) -> Optional[Path]:
    cands: List[Path] = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn == "images.txt":
                cands.append(Path(dp) / fn)
    cands.sort(key=lambda p: (-len(p.parts), str(p)))
    for p in cands:
        try:
            _ = parse_colmap_images_txt(p)
            return p
        except Exception:
            continue
    return None

def infer_sequence_name(out_dir: Path) -> Optional[str]:
    """
    ViPE often outputs a sequence folder under out_dir.
    If there is exactly one subdir (excluding hidden), return it.
    """
    subs = [p for p in out_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if len(subs) == 1:
        return subs[0].name
    return None

def vipe_to_colmap(out_dir: Path, vipe_repo: Path, sequence: Optional[str]) -> None:
    script = vipe_repo / "scripts" / "vipe_to_colmap.py"
    if not script.exists():
        raise FileNotFoundError(f"Cannot find {script}")
    cmd = ["python", str(script), str(out_dir)]
    if sequence is None:
        sequence = infer_sequence_name(out_dir)
    if sequence is not None:
        cmd += ["--sequence", sequence]
    run_cmd(cmd, cwd=vipe_repo)

def parse_colmap_images_txt(images_txt: Path) -> np.ndarray:
    """
    Parse COLMAP images.txt -> camera->world poses (T_wc) (N,4,4).
    COLMAP stores world->cam: qvec(w,x,y,z), tvec.
    Convert:
      R_wc = R_cw^T
      C_w  = -R_wc t_cw
    """
    lines = images_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    entries: List[Tuple[str, np.ndarray]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]
        # skip points2D line
        if i < len(lines):
            i += 1

        R_cw = quat_to_R_wxyz(qw, qx, qy, qz)
        t_cw = np.array([tx, ty, tz], dtype=np.float64)
        R_wc = R_cw.T
        C_w = -R_wc @ t_cw
        entries.append((name, make_T(R_wc, C_w)))

    if not entries:
        raise RuntimeError(f"No valid entries in {images_txt}")

    entries.sort(key=lambda x: x[0])
    return np.stack([e[1] for e in entries], axis=0)

def extract_traj(video: Path, cache_dir: Path, vipe_mode: str, vipe_repo: Optional[Path],
                 pipeline: str, pose_only: bool, try_colmap: bool) -> np.ndarray:
    key = f"{video.name}-{_sha1(str(video.resolve()))}"
    out_dir = cache_dir / "vipe" / key
    ensure_vipe(video, out_dir, vipe_mode, vipe_repo, pipeline, pose_only)

    images_txt = find_images_txt(out_dir)
    if images_txt is None and try_colmap:
        if vipe_repo is None:
            raise ValueError("Need --vipe_repo to run vipe_to_colmap conversion.")
        vipe_to_colmap(out_dir, vipe_repo, sequence=None)
        # vipe_to_colmap writes COLMAP files under a sibling "*_colmap" directory.
        # First look again under the original out_dir, then under the "*_colmap" tree.
        images_txt = find_images_txt(out_dir)
        if images_txt is None:
            colmap_root = out_dir.parent / f"{out_dir.name}_colmap"
            images_txt = find_images_txt(colmap_root)

    if images_txt is None:
        raise RuntimeError(
            f"Could not find COLMAP images.txt under {out_dir}.\n"
            f"Tip: pass --try_vipe_to_colmap and --vipe_repo /path/to/vipe."
        )

    return parse_colmap_images_txt(images_txt)

# -----------------------------
# Action bucketing (optional breakdown)
# -----------------------------

def bucket_for_action(a: str) -> str:
    a = a.lower()
    if a in {"w", "a", "s", "d", "forward", "backward", "left", "right"}:
        return "translation"
    if a in {"cam_left", "cam_right", "cam_up", "cam_down",
             "yaw_left", "yaw_right", "pitch_up", "pitch_down"}:
        return "rotation"
    return "other"

# -----------------------------
# Main eval
# -----------------------------

def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True, help="JSONL episodes")
    ap.add_argument("--models", type=str, required=True, help="Comma-separated model names")
    ap.add_argument("--gen_root", type=str, required=True, help="Generated videos root")
    ap.add_argument("--gen_pattern", type=str, default="{gen_root}/{model}/{episode_id}.mp4",
                    help="Pattern to locate gen video")

    ap.add_argument("--cache_dir", type=str, required=True, help="Cache dir for ViPE outputs")
    ap.add_argument("--out_csv", type=str, required=True)

    # ViPE
    ap.add_argument("--vipe_mode", type=str, default="cli", choices=["cli", "runpy"])
    ap.add_argument("--vipe_repo", type=str, default="", help="Path to ViPE repo (needed for runpy and/or vipe_to_colmap)")
    ap.add_argument("--vipe_pipeline", type=str, default="default")
    ap.add_argument("--pose_only", action="store_true", help="(runpy) set depth_align_model=null")
    ap.add_argument("--try_vipe_to_colmap", action="store_true",
                    help="If images.txt not found, run scripts/vipe_to_colmap.py (requires --vipe_repo)")

    # RPE
    ap.add_argument("--delta", type=int, default=1)
    ap.add_argument("--min_valid_steps", type=int, default=8)
    ap.add_argument("--max_rot_deg", type=float, default=179.0, help="filter absurd rot errors (pose failure)")
    ap.add_argument("--max_trans", type=float, default=1e9, help="filter absurd trans errors (pose failure)")

    # breakdown
    ap.add_argument("--per_action", action="store_true", help="Also output per-action rows (in addition to groups)")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    gen_root = Path(args.gen_root)
    cache_dir = Path(args.cache_dir)
    out_csv = Path(args.out_csv)

    vipe_repo = Path(args.vipe_repo) if args.vipe_repo else None
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    episodes = load_manifest_jsonl(manifest)

    # Aggregation buffers:
    # model -> bucket -> lists
    trans_buf: Dict[str, Dict[str, List[float]]] = {m: {} for m in models}
    rot_buf: Dict[str, Dict[str, List[float]]] = {m: {} for m in models}
    count_buf: Dict[str, Dict[str, int]] = {m: {} for m in models}

    def add(m: str, bucket: str, te: np.ndarray, re: np.ndarray):
        trans_buf[m].setdefault(bucket, [])
        rot_buf[m].setdefault(bucket, [])
        count_buf[m].setdefault(bucket, 0)
        trans_buf[m][bucket].extend(te.tolist())
        rot_buf[m][bucket].extend(re.tolist())
        count_buf[m][bucket] += int(te.size)

    # Helper to materialize current buffers into CSV (for periodic flushing)
    def dump_csv() -> None:
        def stats(xs: List[float]) -> Tuple[float, float]:
            if not xs:
                return float("nan"), float("nan")
            arr = np.array(xs, dtype=np.float64)
            return float(arr.mean()), float(np.median(arr))

        rows: List[Dict[str, object]] = []
        for m in models:
            buckets = sorted(trans_buf[m].keys(), key=lambda b: (b != "__overall__", b))
            for b in buckets:
                tmean, tmed = stats(trans_buf[m][b])
                rmean, rmed = stats(rot_buf[m][b])
                rows.append({
                    "model": m,
                    "bucket": b,
                    "count": count_buf[m].get(b, 0),
                    "rpe_trans_mean": tmean,
                    "rpe_trans_median": tmed,
                    "rpe_rot_mean_deg": rmean,
                    "rpe_rot_median_deg": rmed,
                })

        write_csv(rows, out_csv)

    flush_every = 5  # episodes

    for ep_idx, ep in enumerate(episodes):
        if not ep.gt_video.exists():
            print(f"[skip] GT video does not exist for episode {ep.episode_id}: {ep.gt_video}")
            continue

        # Canonical (GT) trajectory from GT video
        try:
            gt_T = extract_traj(
                ep.gt_video, cache_dir,
                vipe_mode=args.vipe_mode, vipe_repo=vipe_repo,
                pipeline=args.vipe_pipeline, pose_only=args.pose_only,
                try_colmap=args.try_vipe_to_colmap,
            )
        except Exception as e:
            print(f"[skip] ViPE/poses failed on GT for episode {ep.episode_id}: {e}")
            continue

        for m in models:
            gen_path = Path(args.gen_pattern.format(
                gen_root=str(gen_root), model=m, episode_id=ep.episode_id
            ))
            if not gen_path.exists():
                # Fallback: try to glob any mp4 starting with the episode_id
                base_dir = gen_root
                if "{model}" in args.gen_pattern:
                    base_dir = base_dir / m
                candidates = sorted(base_dir.glob(f"{ep.episode_id}*.mp4")) if base_dir.exists() else []
                if len(candidates) == 1:
                    print(f"[info] Using fallback gen video for episode {ep.episode_id}, model {m}: {candidates[0]}")
                    gen_path = candidates[0]
                elif len(candidates) > 1:
                    print(f"[skip] Multiple candidate gen videos for episode {ep.episode_id}, model {m} under {base_dir}, please disambiguate.")
                    continue
                else:
                    print(f"[skip] Gen video does not exist for episode {ep.episode_id}, model {m}: {gen_path}")
                    continue

            try:
                gen_T = extract_traj(
                    gen_path, cache_dir,
                    vipe_mode=args.vipe_mode, vipe_repo=vipe_repo,
                    pipeline=args.vipe_pipeline, pose_only=args.pose_only,
                    try_colmap=args.try_vipe_to_colmap,
                )
            except Exception as e:
                print(f"[skip] ViPE/poses failed on GEN for episode {ep.episode_id}, model {m}: {e}")
                continue

            # Align lengths to actions (+delta poses)
            n_steps = min(len(ep.actions), len(gt_T) - args.delta, len(gen_T) - args.delta)
            if n_steps <= 0:
                print(f"[skip] Not enough steps after alignment for episode {ep.episode_id}, model {m}: n_steps={n_steps}")
                continue
            gt_use = gt_T[:(n_steps + args.delta)]
            gen_use = gen_T[:(n_steps + args.delta)]
            actions = ep.actions[:n_steps]

            # Sim(3) align gen->gt (positions)
            s, R, t = umeyama_sim3(gen_use[:, :3, 3], gt_use[:, :3, 3])
            gen_aligned = apply_sim3_to_poses(gen_use, s, R, t)

            # RPE
            te, re = compute_rpe(gt_use, gen_aligned, delta=args.delta)

            # Filter obvious pose failures
            valid = np.ones_like(te, dtype=bool)
            if args.max_rot_deg > 0:
                valid &= (re <= args.max_rot_deg)
            if args.max_trans > 0:
                valid &= (te <= args.max_trans)

            te = te[valid]
            re = re[valid]
            actions_valid = [a for (a, ok) in zip(actions, valid.tolist()) if ok]

            if te.size < args.min_valid_steps:
                print(f"[skip] Too few valid RPE samples for episode {ep.episode_id}, model {m}: te.size={te.size}, min_valid_steps={args.min_valid_steps}")
                continue

            # Overall
            add(m, "__overall__", te, re)

            # Group buckets (translation/rotation/other)
            groups = [bucket_for_action(a) for a in actions_valid]
            for grp in {"translation", "rotation", "other"}:
                idx = [i for i, g in enumerate(groups) if g == grp]
                if idx:
                    add(m, grp, te[idx], re[idx])

            # Optional per-action breakdown
            if args.per_action:
                uniq = sorted(set(actions_valid))
                for a in uniq:
                    idx = [i for i, aa in enumerate(actions_valid) if aa == a]
                    add(m, f"act:{a}", te[idx], re[idx])
        # Periodically flush partial results to CSV
        if (ep_idx + 1) % flush_every == 0:
            dump_csv()

    # Final write after processing all episodes
    dump_csv()
    print(f"[done] wrote {out_csv}")

if __name__ == "__main__":
    main()
