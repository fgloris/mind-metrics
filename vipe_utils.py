from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import subprocess
import numpy as np
import os

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

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def find_images_txt(root: Path) -> Optional[Path]:
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn == "images.txt":
                return (Path(dp) / fn)


def vipe_to_colmap(out_dir: Path, vipe_repo: Path) -> None:
    script = vipe_repo / "scripts" / "vipe_to_colmap.py"
    if not script.exists():
        raise FileNotFoundError(f"Cannot find {script}")
    cmd = ["python", str(script), str(out_dir)]
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

def extract_traj(video: Path, cache_dir: Path, pipeline: str = "default") -> np.ndarray:
    video = video.resolve()
    cache_dir = cache_dir.resolve()
    key = f"{video.name}-{_sha1(str(video))}"
    out_dir = cache_dir / "vipe" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(["vipe", "infer", str(video), "--output", str(out_dir), "--pipeline", pipeline])

    vipe_to_colmap(out_dir, Path("./vipe"))
    images_txt = find_images_txt(out_dir.parent / f"{out_dir.name}_colmap")

    if images_txt is None:
        raise RuntimeError(
            f"Could not find COLMAP images.txt under {out_dir}.\n"
            f"Tip: pass --try_vipe_to_colmap and --vipe_repo /path/to/vipe."
        )

    return parse_colmap_images_txt(images_txt)

def bucket_for_action(a: str) -> str:
    a = a.lower()
    if a in {"w", "a", "s", "d", "forward", "backward", "left", "right"}:
        return "translation"
    if a in {"cam_left", "cam_right", "cam_up", "cam_down",
             "yaw_left", "yaw_right", "pitch_up", "pitch_down"}:
        return "rotation"
    return "other"