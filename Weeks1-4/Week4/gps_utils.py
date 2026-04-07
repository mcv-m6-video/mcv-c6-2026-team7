import os
import math
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────

def parse_homography_file(calib_path: str):
    """
    Read a calibration.txt with AI City format:

        Homography matrix: a b c;d e f;g h i
        Reprojection error: X.XX
    """
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(f"calibration.txt not found: {calib_path}")

    H = None
    reproj_error = None

    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Homography matrix:"):
                vals_str = line.split(":", 1)[1].strip()
                rows = vals_str.split(";")
                H = np.array(
                    [[float(v) for v in r.split()] for r in rows],
                    dtype=np.float64,
                )
            elif line.startswith("Reprojection error:"):
                reproj_error = float(line.split(":", 1)[1].strip())

    if H is None:
        raise ValueError(f"'Homography matrix' not found in {calib_path}")
    if reproj_error is None:
        reproj_error = float("nan")

    return H, reproj_error


def load_homographies(seq_dir: str, cam_ids: list) -> dict:
    """Load calibration.txt for each camera in `cam_ids`."""
    homos = {}
    for cam_id in cam_ids:
        cam_name   = f"c{cam_id:03d}"
        calib_path = os.path.join(seq_dir, cam_name, "calibration.txt")
        try:
            H, err = parse_homography_file(calib_path)
            homos[cam_id] = (H, err)
            print(f"  [GPS] {cam_name}: homography loaded (reproj_error={err:.3f} px)")
        except FileNotFoundError:
            print(f"  [GPS] {cam_name}: calibration.txt not found — GPS disabled for this camera.")
            homos[cam_id] = (None, None)
        except Exception as exc:
            print(f"  [GPS] {cam_name}: error reading calibration — {exc}")
            homos[cam_id] = (None, None)

    n_valid = sum(1 for v in homos.values() if v[0] is not None)
    print(f"  [GPS] {n_valid}/{len(cam_ids)} cameras with valid homography.\n")
    return homos


# ─────────────────────────────────────────────
# Image → ground-plane projection
# ─────────────────────────────────────────────

def pixel_to_world(H: np.ndarray, px: float, py: float):
    """Project an image point (px, py) to the ground plane using H."""
    pt = H @ np.array([px, py, 1.0], dtype=np.float64)
    return float(pt[0] / pt[2]), float(pt[1] / pt[2])


def _car_ground_point(row: pd.Series, H: np.ndarray):
    """
    Project the best-estimate ground contact point of a car to world coords.

    WHY NOT THE FOOT POINT (cx, y2)?
    The homography maps the ground plane. For pedestrians, the bottom-center
    of the bbox (the feet) actually touches the ground — so (cx, y2) is correct.
    For cars viewed from a typical surveillance angle, y2 is the bottom edge of
    the nearest visible bumper, which is elevated and varies with car size and
    viewing angle. The geometric center of the bbox (cx, cy) is a far more
    stable estimator of the car's ground footprint centroid across varying
    distances and angles, because perspective distortion affects the top and
    bottom of the bbox nearly symmetrically.
    """
    cx = (row["x1"] + row["x2"]) / 2.0
    cy = (row["y1"] + row["y2"]) / 2.0   # bbox center, not bottom edge
    return pixel_to_world(H, cx, cy)


def tracklet_first_world_pos(tdf: pd.DataFrame, H: np.ndarray):
    """
    Return the ground-plane position of the FIRST frame of the tracklet
    (sorted by timestamp, then frame_id as tiebreaker) and its timestamp.

    Using the first/last point rather than a centroid captures the real
    camera entry/exit moment for temporal gating.
    """
    row = tdf.sort_values(["timestamp", "frame_id"]).iloc[0]
    return _car_ground_point(row, H), float(row["timestamp"])


def tracklet_last_world_pos(tdf: pd.DataFrame, H: np.ndarray):
    """
    Return the ground-plane position of the LAST frame of the tracklet
    and its timestamp.
    """
    row = tdf.sort_values(["timestamp", "frame_id"]).iloc[-1]
    return _car_ground_point(row, H), float(row["timestamp"])


# ─────────────────────────────────────────────
# World-coordinate scale auto-estimation
# ─────────────────────────────────────────────

def estimate_world_scale(tracklets: list) -> float:
    """
    Automatically estimate how many world-coordinate units equal one metre,
    by analysing intra-tracklet motion.

    Method: for every consecutive detection pair within a single tracklet we
    compute the world-space displacement and the time gap. This gives us an
    observed speed in world-units/second. Cars in mixed urban traffic have a
    characteristic speed distribution; we assume the 60th percentile of all
    observed speeds corresponds to ~7 m/s (≈25 km/h, a reasonable median for
    urban surveillance sequences). The scale factor is then:

        scale (world_units / metre) = observed_60pct_speed / 7.0

    The 60th percentile is used rather than the median because stopped/idling
    cars at junctions pull the median down, while the 60th percentile better
    represents cars in active motion.

    Falls back to 1.0 (no rescaling) if there are too few samples or if the
    estimated motion is near-zero (e.g. a sequence of parked cars).

    Returns:
        scale : float — world units per metre
    """
    ASSUMED_MEDIAN_SPEED_MPS = 7.0

    speeds_world = []

    for t in tracklets:
        H = t.get("_H")
        if H is None:
            continue
        tdf = t["df"].sort_values(["timestamp", "frame_id"])
        if len(tdf) < 2:
            continue

        prev_wx, prev_wy, prev_ts = None, None, None
        for _, row in tdf.iterrows():
            cx = (row["x1"] + row["x2"]) / 2.0
            cy = (row["y1"] + row["y2"]) / 2.0
            wx, wy = pixel_to_world(H, cx, cy)
            ts = float(row["timestamp"])

            if prev_ts is not None:
                dt = ts - prev_ts
                if dt > 0.05:   # skip duplicate / near-duplicate frames
                    dist = math.hypot(wx - prev_wx, wy - prev_wy)
                    speeds_world.append(dist / dt)

            prev_wx, prev_wy, prev_ts = wx, wy, ts

    if len(speeds_world) < 10:
        print("  [GPS] Scale estimation: too few motion samples — using scale=1.0 (no rescaling)")
        return 1.0

    p60 = float(np.percentile(speeds_world, 60))

    if p60 < 1e-3:
        print("  [GPS] Scale estimation: near-zero motion detected — using scale=1.0")
        return 1.0

    scale = p60 / ASSUMED_MEDIAN_SPEED_MPS
    print(f"  [GPS] Scale estimation: 60th-pct intra-tracklet speed = {p60:.3f} u/s  "
          f"→  scale = {scale:.4f} world-units/metre  "
          f"(assumed median car speed = {ASSUMED_MEDIAN_SPEED_MPS} m/s)")
    return scale


# ─────────────────────────────────────────────
# Spatio-temporal gate
# ─────────────────────────────────────────────

def build_spatiotemporal_gate(tracklets: list,
                               max_speed_mps: float = 30.0,
                               min_dt_s: float = 0.5,
                               world_scale: float = 1.0,
                               reproj_error_threshold: float = 15.0) -> np.ndarray:
    """
    Build a boolean mask (n × n) where:
        True  → pair is allowed for merging
        False → pair is physically impossible and must be blocked

    Three conditions are applied to every cross-camera pair (i, j):

    ① TEMPORAL OVERLAP
      If the two tracklets are active at the same time in different cameras
      they could be the same car in overlapping FOVs. We do NOT block them,
      but we also skip the speed gate (dt ≈ 0 → infinite implied speed).
      These pairs fall back to the colour gate only.

    ② SPEED GATE
      For non-overlapping pairs with dt >= min_dt_s, we compute the implied
      travel speed from the exit position of the earlier tracklet to the entry
      position of the later one. If speed > max_speed_mps the pair is blocked.
      world_scale converts world units → metres for the comparison.

    ③ REPROJECTION-ERROR RELAXATION
      Cameras with high reprojection error have unreliable homographies.
      For pairs where either camera's reprojection error exceeds
      reproj_error_threshold, the speed limit is relaxed proportionally:

          effective_limit = max_speed_mps × (worst_error / threshold)

      This avoids hard-blocking valid matches due to inaccurate projections,
      while still catching grossly impossible pairs.

    Args:
        tracklets              : list of tracklet dicts; must contain _reproj_error
        max_speed_mps          : max physically plausible car speed in m/s
        min_dt_s               : minimum time gap (s) before applying speed gate
        world_scale            : world-units per metre, from estimate_world_scale()
        reproj_error_threshold : reproj error (px) above which speed limit is relaxed
    """
    n = len(tracklets)
    allowed = np.ones((n, n), dtype=bool)
    n_blocked_speed = 0

    # Speed limit in world units / second
    max_speed_world = max_speed_mps * world_scale

    for i in range(n):
        ti = tracklets[i]
        if ti["cam_id"] is None:
            continue
        first_i = ti.get("world_first")
        last_i  = ti.get("world_last")
        if first_i is None or last_i is None:
            continue

        (wx_last_i,  wy_last_i),  t_end_i   = last_i
        (wx_first_i, wy_first_i), t_start_i = first_i
        err_i = ti.get("_reproj_error") or 0.0

        for j in range(i + 1, n):
            tj = tracklets[j]
            if ti["cam_id"] == tj["cam_id"]:
                continue   # same camera — handled by distance matrix

            first_j = tj.get("world_first")
            last_j  = tj.get("world_last")
            if first_j is None or last_j is None:
                continue

            (wx_last_j,  wy_last_j),  t_end_j   = last_j
            (wx_first_j, wy_first_j), t_start_j = first_j
            err_j = tj.get("_reproj_error") or 0.0

            # ③ Reprojection-error relaxation — computed first, used by both branches
            worst_err = max(err_i, err_j)
            if worst_err > reproj_error_threshold:
                relaxation = worst_err / reproj_error_threshold
                effective_max = max_speed_world * relaxation
            else:
                effective_max = max_speed_world

            # ① Temporal overlap — interpolate positions at overlap midpoint
            # Timestamps are globally synchronised across cameras, so two tracklets
            # active at the same time in different cameras that are far apart in
            # world space cannot be the same car. We use 1 second of travel distance
            # as the spatial budget for simultaneous tracklets.
            if t_start_i < t_end_j and t_start_j < t_end_i:
                overlap_mid = (max(t_start_i, t_start_j) + min(t_end_i, t_end_j)) / 2.0
                alpha_i = (overlap_mid - t_start_i) / max(t_end_i - t_start_i, 1e-6)
                alpha_i = max(0.0, min(1.0, alpha_i))
                wx_i = wx_first_i + alpha_i * (wx_last_i - wx_first_i)
                wy_i = wy_first_i + alpha_i * (wy_last_i - wy_first_i)
                alpha_j = (overlap_mid - t_start_j) / max(t_end_j - t_start_j, 1e-6)
                alpha_j = max(0.0, min(1.0, alpha_j))
                wx_j = wx_first_j + alpha_j * (wx_last_j - wx_first_j)
                wy_j = wy_first_j + alpha_j * (wy_last_j - wy_first_j)
                sim_dist = math.hypot(wx_i - wx_j, wy_i - wy_j)
                if sim_dist > effective_max * 1.0:  # 1-second spatial budget
                    allowed[i, j] = False
                    allowed[j, i] = False
                    n_blocked_speed += 1
                continue

            # Determine which tracklet ends first and compute the gap
            if t_end_i <= t_end_j:
                dt   = t_start_j - t_end_i
                wx_a, wy_a = wx_last_i,  wy_last_i    # exit of i
                wx_b, wy_b = wx_first_j, wy_first_j   # entry of j
            else:
                dt   = t_start_i - t_end_j
                wx_a, wy_a = wx_last_j,  wy_last_j    # exit of j
                wx_b, wy_b = wx_first_i, wy_first_i   # entry of i

            # Time gap too small — skip (avoids fp division noise)
            if dt < min_dt_s:
                continue

            # ② Speed gate
            dist  = math.hypot(wx_a - wx_b, wy_a - wy_b)
            speed = dist / dt

            if speed > effective_max:
                allowed[i, j] = False
                allowed[j, i] = False
                n_blocked_speed += 1

    print(f"  [GPS] Spatio-temporal gate: {n_blocked_speed} cross-cam pairs blocked "
          f"(max_speed={max_speed_mps} m/s, scale={world_scale:.4f} u/m, "
          f"reproj_threshold={reproj_error_threshold} px)")
    return allowed


# ─────────────────────────────────────────────
# Utility: per-row world coordinate computation
# ─────────────────────────────────────────────

def row_to_world(row: pd.Series, H) -> tuple:
    """
    Given a DataFrame row and a homography H (or None),
    return (xw, yw) rounded to 2 decimal places, or (-1, -1) if H is None.
    Uses the car-correct center projection (not foot point).
    """
    if H is None:
        return -1, -1
    cx = (row["x1"] + row["x2"]) / 2
    cy = (row["y1"] + row["y2"]) / 2   # center, not bottom edge
    xw, yw = pixel_to_world(H, cx, cy)
    return round(xw, 2), round(yw, 2)