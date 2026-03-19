"""
camera_time_windows.py
======================
Computes per-camera-pair expected travel times and tolerance windows
from homography calibration files.

For every pair of cameras in a sequence it:
  1. Projects the image centre  (W/2, H/2)  to world-plane coords
     using the camera's homography matrix.
  2. Computes the Euclidean distance between each pair of world-plane
     centres.
  3. Divides by `avg_speed` (world-units / second, same units as the
     homography output) to get the expected travel time T[i,j].
  4. Saves a JSON file with the expected time and the tolerance window
     for every ordered camera pair.

The JSON is consumed by MTMC_gps.py (build_timewindow_gate) to hard-block
cross-camera tracklet pairs whose timing is inconsistent with the expected
transit time.

Usage
-----
Single sequence:
    python camera_time_windows.py S01 \\
        --seq-root AI_CITY_CHALLENGE_2022_TRAIN/train \\
        --img-w 1920 --img-h 1080 \\
        --avg-speed 5.0 \\
        --window 10.0 \\
        --output results/S01_time_windows.json

All sequences found under --seq-root:
    python camera_time_windows.py --all \\
        --seq-root AI_CITY_CHALLENGE_2022_TRAIN/train \\
        --img-w 1920 --img-h 1080 \\
        --avg-speed 5.0 \\
        --window 10.0 \\
        --out-dir results/

Arguments
---------
    sequence        Sequence name, e.g. S01.  Omit when using --all.
    --all           Process every SXX directory found under --seq-root.
                    Mutually exclusive with positional sequence argument.
    --seq-root      Directory that contains <sequence>/cXXX/calibration.txt
    --img-w         Frame width  in pixels  (default: 1920)
    --img-h         Frame height in pixels  (default: 1080)
    --avg-speed     Expected average speed in homography world-units per
                    second.  Match the unit system of your calibration:
                      • metres/s  if H maps pixels → metres  (~5–15 for cars)
                      • feet/s    if H maps pixels → feet    (~15–50 for cars)
    --window        Tolerance window (seconds) added symmetrically around
                    the expected travel time.  A generous starting value is
                    half the expected travel time, or a fixed 5–15 s.
    --output        Output JSON path (single-sequence mode only).  Defaults to
                    <seq-root>/<sequence>_time_windows.json
    --out-dir       Output directory for batch mode (--all).  Each sequence
                    writes its own <sequence>_time_windows.json here.
                    Defaults to --seq-root.

Output JSON schema
------------------
{
    "meta": {
        "sequence":  "S01",
        "avg_speed": 5.0,
        "window":    10.0,
        "img_w":     1920,
        "img_h":     1080
    },
    "cameras": {
        "1": [wx, wy],   // world-plane centre of camera c001
        "2": [wx, wy],
        ...
    },
    "pairs": {
        "1__2": {"dist": 42.3, "expected_t": 8.46,  "t_min": 0.0,  "t_max": 18.46},
        "2__1": {"dist": 42.3, "expected_t": 8.46,  "t_min": 0.0,  "t_max": 18.46},
        ...
    }
}

    t_min = max(0, expected_t - window)
    t_max = expected_t + window
"""

import os
import sys
import math
import json
import argparse
import glob

import numpy as np

# Re-use parsing utilities from gps_utils so there is a single source of truth.
from gps_utils import load_homographies, pixel_to_world


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Build camera-pair time windows from homography calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Mode selection ───────────────────────────────────────────────────────
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true",
                      help="Process every SXX directory found under --seq-root. "
                           "Cannot be used together with the positional sequence argument.")

    p.add_argument("sequence", type=str, nargs="?", default=None,
                   help="Sequence name, e.g. S01.  Omit when using --all.")

    # ── Shared options ───────────────────────────────────────────────────────
    p.add_argument("--seq-root", type=str,
                   default="AI_CITY_CHALLENGE_2022_TRAIN/train",
                   help="Directory that contains <sequence>/cXXX/calibration.txt")
    p.add_argument("--img-w", type=int, default=1920,
                   help="Frame width in pixels")
    p.add_argument("--img-h", type=int, default=1080,
                   help="Frame height in pixels")
    p.add_argument("--avg-speed", type=float, default=5.0,
                   help="Expected average speed in homography world-units per second")
    p.add_argument("--window", type=float, default=10.0,
                   help="Symmetric tolerance window in seconds around the expected travel time")

    # ── Output options ───────────────────────────────────────────────────────
    p.add_argument("--output", type=str, default=None,
                   help="Output JSON path (single-sequence mode). "
                        "Defaults to <seq-root>/<sequence>_time_windows.json")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory for batch mode (--all). "
                        "Each sequence writes <sequence>_time_windows.json here. "
                        "Defaults to --seq-root.")

    args = p.parse_args()

    # ── Validate mode ────────────────────────────────────────────────────────
    if args.all and args.sequence:
        p.error("--all and a positional sequence argument are mutually exclusive. "
                "Use one or the other.")
    if not args.all and not args.sequence:
        p.error("Provide either a sequence name (e.g. S01) or --all to process "
                "every sequence under --seq-root.")
    if args.output and args.all:
        p.error("--output is for single-sequence mode only. "
                "Use --out-dir to set the output directory in --all mode.")

    return args


# ─────────────────────────────────────────────
# Camera discovery
# ─────────────────────────────────────────────

def discover_cam_ids(seq_dir: str, fatal: bool = True) -> list:
    """
    Return a sorted list of integer camera IDs found under seq_dir.
    A camera directory is any folder matching the pattern cXXX.

    Parameters
    ----------
    fatal : bool
        If True (default), call sys.exit() when no cameras are found.
        Set to False in batch mode so a missing/empty sequence is skipped
        with a warning instead of aborting the whole run.
    """
    pattern = os.path.join(seq_dir, "c[0-9][0-9][0-9]")
    cam_dirs = sorted(glob.glob(pattern))
    if not cam_dirs:
        msg = f"[ERROR] No cXXX directories found in {seq_dir}"
        if fatal:
            sys.exit(msg)
        print(f"  [WARN] {msg} — skipping.")
        return []
    cam_ids = [int(os.path.basename(d)[1:]) for d in cam_dirs]
    print(f"  Found {len(cam_ids)} camera directories: "
          f"{[f'c{c:03d}' for c in cam_ids]}")
    return cam_ids


def discover_sequences(seq_root: str) -> list:
    """
    Return a sorted list of sequence names (e.g. ['S01', 'S03', 'S04'])
    found directly under seq_root.  A sequence directory is any folder
    whose name matches S + one or more digits (case-insensitive).
    """
    entries = sorted(os.listdir(seq_root))
    sequences = [
        e for e in entries
        if os.path.isdir(os.path.join(seq_root, e))
        and len(e) >= 2
        and e[0].upper() == "S"
        and e[1:].isdigit()
    ]
    if not sequences:
        sys.exit(f"[ERROR] No sequence directories (SXX) found under {seq_root}")
    print(f"  Found {len(sequences)} sequence(s): {sequences}")
    return sequences


# ─────────────────────────────────────────────
# World-plane camera centres
# ─────────────────────────────────────────────

def compute_camera_world_centres(homographies: dict,
                                  img_w: int,
                                  img_h: int) -> dict:
    """
    Project the image centre (img_w/2, img_h/2) to world-plane coordinates
    for each camera that has a valid homography.

    Returns
    -------
    dict  cam_id (int) → (wx, wy)  world-plane centre
          Cameras without a valid homography are omitted with a warning.
    """
    cx = img_w / 2.0
    cy = img_h / 2.0

    centres = {}
    for cam_id, (H, err) in homographies.items():
        if H is None:
            print(f"  [WARN] c{cam_id:03d}: no homography — skipped.")
            continue
        wx, wy = pixel_to_world(H, cx, cy)
        centres[cam_id] = (wx, wy)
        print(f"  c{cam_id:03d}: image centre ({cx:.0f}, {cy:.0f}) "
              f"→ world ({wx:.2f}, {wy:.2f})")

    return centres


# ─────────────────────────────────────────────
# Distance & time-window matrix
# ─────────────────────────────────────────────

def build_time_windows(centres: dict,
                        avg_speed: float,
                        window: float) -> dict:
    """
    For every ordered camera pair (i, j) with i ≠ j, compute:
        dist       Euclidean distance between world-plane centres
        expected_t dist / avg_speed  (seconds)
        t_min      max(0, expected_t - window)
        t_max      expected_t + window

    Returns a dict keyed by  "<cam_i>__<cam_j>".
    """
    pairs = {}
    cam_ids = sorted(centres.keys())

    for i in cam_ids:
        for j in cam_ids:
            if i == j:
                continue
            wx_i, wy_i = centres[i]
            wx_j, wy_j = centres[j]
            dist = math.hypot(wx_i - wx_j, wy_i - wy_j)
            expected_t = dist / avg_speed if avg_speed > 0 else float("inf")
            t_min = max(0.0, expected_t - window)
            t_max = expected_t + window

            key = f"{i}__{j}"
            pairs[key] = {
                "dist":       round(dist, 4),
                "expected_t": round(expected_t, 4),
                "t_min":      round(t_min, 4),
                "t_max":      round(t_max, 4),
            }

    return pairs


# ─────────────────────────────────────────────
# Pretty-print summary table
# ─────────────────────────────────────────────

def print_summary(pairs: dict, cam_ids: list):
    """
    Print a compact table of expected travel times between camera pairs.
    """
    cam_ids = sorted(cam_ids)
    n = len(cam_ids)

    col_w = 12
    header = f"{'':>6}" + "".join(f"  c{c:03d}{'':>{col_w - 5}}" for c in cam_ids)
    print("\n  Expected travel times (seconds)  [avg_speed units/s]")
    print("  " + "─" * (6 + n * col_w))
    print("  " + header)
    print("  " + "─" * (6 + n * col_w))

    for i in cam_ids:
        row = f"  c{i:03d} |"
        for j in cam_ids:
            if i == j:
                row += f"{'—':>{col_w}}"
            else:
                t = pairs[f"{i}__{j}"]["expected_t"]
                row += f"{t:>{col_w}.2f}"
        print(row)

    print("  " + "─" * (6 + n * col_w))

    print("\n  Window bounds  [t_min, t_max]  in seconds")
    print("  " + "─" * 55)
    for key, v in pairs.items():
        i, j = key.split("__")
        print(f"  c{int(i):03d} → c{int(j):03d} :  "
              f"dist={v['dist']:>8.2f}  |  "
              f"expected={v['expected_t']:>7.2f} s  |  "
              f"window=[{v['t_min']:.2f}, {v['t_max']:.2f}] s")
    print()


# ─────────────────────────────────────────────
# JSON I/O
# ─────────────────────────────────────────────

def save_json(sequence: str,
              avg_speed: float,
              window: float,
              img_w: int,
              img_h: int,
              centres: dict,
              pairs: dict,
              out_path: str):
    """
    Serialise the full result to a JSON file consumed by MTMC_gps.py.
    """
    payload = {
        "meta": {
            "sequence":  sequence,
            "avg_speed": avg_speed,
            "window":    window,
            "img_w":     img_w,
            "img_h":     img_h,
        },
        # Store centres as  str(cam_id) → [wx, wy]  for JSON compatibility
        "cameras": {
            str(cam_id): list(pos) for cam_id, pos in sorted(centres.items())
        },
        "pairs": pairs,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"  Saved time-window JSON → {out_path}")
    print(f"  Pairs stored: {len(pairs)}\n")


def load_time_windows(json_path: str) -> dict:
    """
    Load a previously saved time-windows JSON file.

    Returns the full payload dict (keys: meta, cameras, pairs).
    Raises FileNotFoundError if the file does not exist.

    This function is imported by MTMC_gps.py.
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"[ERROR] Time-windows file not found: {json_path}\n"
            "  Run camera_time_windows.py first to generate it."
        )
    with open(json_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Gate function  (imported by MTMC_gps.py)
# ─────────────────────────────────────────────

def build_timewindow_gate(tracklets: list, json_path: str) -> np.ndarray:
    """
    Build an (n × n) boolean mask where:
        True  → pair is within the expected transit-time window
        False → pair is blocked (timing inconsistent with camera geometry)

    The mask is AND-ed with the speed gate in MTMC_gps.build_distance_matrix.

    Parameters
    ----------
    tracklets : list of dicts
        Each dict must contain:
          - cam_id       (int)
          - world_first  ((wx, wy), t_start)  from gps_utils
          - world_last   ((wx, wy), t_end)    from gps_utils
    json_path : str
        Path to the JSON produced by this script.

    Returns
    -------
    np.ndarray  shape (n, n), dtype bool
    """
    data   = load_time_windows(json_path)
    pairs  = data["pairs"]

    n        = len(tracklets)
    allowed  = np.ones((n, n), dtype=bool)
    n_blocked = 0

    for i, ti in enumerate(tracklets):
        last_i = ti.get("world_last")
        if last_i is None:
            continue
        (_, _), t_end_i = last_i
        if t_end_i is None:
            continue

        for j, tj in enumerate(tracklets):
            if i >= j:
                continue
            if ti["cam_id"] == tj["cam_id"]:
                continue  # same-cam pairs handled elsewhere

            first_j = tj.get("world_first")
            last_j  = tj.get("world_last")
            if first_j is None or last_j is None:
                continue

            (_, _), t_start_j = first_j
            (_, _), t_end_j   = last_j
            if t_start_j is None or t_end_j is None:
                continue

            # ── Check both directions:  i→j  and  j→i ──────────────────
            # Direction i→j: i ends before j starts
            _check_and_block(
                i, j, ti["cam_id"], tj["cam_id"],
                t_end_i, t_start_j,
                pairs, allowed,
            )

            # Direction j→i: j ends before i starts
            (_, _), t_end_j2   = tj.get("world_last")
            (_, _), t_start_i2 = ti.get("world_first")
            if t_end_j2 is not None and t_start_i2 is not None:
                _check_and_block(
                    j, i, tj["cam_id"], ti["cam_id"],
                    t_end_j2, t_start_i2,
                    pairs, allowed,
                )

            # If either direction blocked the pair, mark symmetric entry too
            if not allowed[i, j]:
                allowed[j, i] = False
                n_blocked += 1

    print(f"  [TimeWindow] Gate: {n_blocked} cross-cam pairs blocked "
          f"(window loaded from {json_path})")
    return allowed


def _check_and_block(i, j, cam_i, cam_j,
                     t_end_src, t_start_dst,
                     pairs, allowed):
    """
    Block the (i, j) pair if the observed transit time
    t_start_dst - t_end_src falls outside the [t_min, t_max]
    window for the cam_i → cam_j pair.

    Only applied when t_end_src < t_start_dst  (src ends before dst starts).
    Overlapping tracklets are left unrestricted.
    """
    if t_end_src >= t_start_dst:
        return  # overlapping in time — no restriction

    key = f"{cam_i}__{cam_j}"
    if key not in pairs:
        return  # no calibration data for this pair — leave unrestricted

    dt     = t_start_dst - t_end_src   # observed transit time (s)
    t_min  = pairs[key]["t_min"]
    t_max  = pairs[key]["t_max"]

    if not (t_min <= dt <= t_max):
        allowed[i, j] = False


# ─────────────────────────────────────────────
# Per-sequence processing (shared by both modes)
# ─────────────────────────────────────────────

def process_sequence(sequence: str,
                     seq_root: str,
                     img_w: int,
                     img_h: int,
                     avg_speed: float,
                     window: float,
                     out_path: str,
                     fatal: bool = True) -> bool:
    """
    Run the full time-window computation for a single sequence and save
    the result to out_path.

    Parameters
    ----------
    fatal : bool
        Passed to discover_cam_ids.  True in single-sequence mode
        (errors abort the process); False in batch mode (errors are
        reported and the sequence is skipped).

    Returns True on success, False if the sequence was skipped.
    """
    seq_dir = os.path.join(seq_root, sequence)
    if not os.path.isdir(seq_dir):
        msg = f"[ERROR] Sequence directory not found: {seq_dir}"
        if fatal:
            sys.exit(msg)
        print(f"  [WARN] {msg} — skipping.")
        return False

    print(f"\n  Sequence  : {sequence}")
    print(f"  Seq dir   : {seq_dir}")
    print(f"  Img size  : {img_w} × {img_h}")
    print(f"  Avg speed : {avg_speed} world-units/s")
    print(f"  Window    : ± {window} s")

    # 1. Discover cameras
    cam_ids = discover_cam_ids(seq_dir, fatal=fatal)
    if not cam_ids:
        return False

    # 2. Load homographies
    homographies = load_homographies(seq_dir, cam_ids)

    # 3. Project image centres to world plane
    centres = compute_camera_world_centres(homographies, img_w, img_h)
    if len(centres) < 2:
        msg = (f"[WARN] {sequence}: fewer than 2 cameras have valid homographies "
               "— cannot build inter-camera time windows, skipping.")
        if fatal:
            sys.exit("[ERROR]" + msg[6:])
        print(msg)
        return False

    # 4. Build windows & save
    pairs = build_time_windows(centres, avg_speed, window)
    print_summary(pairs, list(centres.keys()))
    save_json(
        sequence=sequence,
        avg_speed=avg_speed,
        window=window,
        img_w=img_w,
        img_h=img_h,
        centres=centres,
        pairs=pairs,
        out_path=out_path,
    )
    return True


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    if args.all:
        # ── Batch mode: process every sequence under seq_root ────────────
        print("\n" + "═" * 60)
        print(f"  Mode      : BATCH (--all)")
        print(f"  Seq root  : {args.seq_root}")
        print(f"  Img size  : {args.img_w} × {args.img_h}")
        print(f"  Avg speed : {args.avg_speed} world-units/s")
        print(f"  Window    : ± {args.window} s")
        out_dir = args.out_dir or args.seq_root
        print(f"  Output dir: {out_dir}")
        print("═" * 60)

        sequences = discover_sequences(args.seq_root)
        os.makedirs(out_dir, exist_ok=True)

        ok, skipped = [], []
        for seq in sequences:
            print("\n" + "─" * 60)
            print(f"  Processing {seq}…")
            print("─" * 60)
            out_path = os.path.join(out_dir, f"{seq}_time_windows.json")
            success = process_sequence(
                sequence=seq,
                seq_root=args.seq_root,
                img_w=args.img_w,
                img_h=args.img_h,
                avg_speed=args.avg_speed,
                window=args.window,
                out_path=out_path,
                fatal=False,          # non-fatal: skip bad sequences
            )
            (ok if success else skipped).append(seq)

        print("\n" + "═" * 60)
        print(f"  Batch complete.")
        print(f"  Processed : {len(ok)}  — {ok}")
        if skipped:
            print(f"  Skipped   : {len(skipped)}  — {skipped}")
        print("═" * 60 + "\n")

    else:
        # ── Single-sequence mode ─────────────────────────────────────────
        out_path = args.output or os.path.join(
            args.seq_root, f"{args.sequence}_time_windows.json"
        )

        print("\n" + "═" * 60)
        print(f"  Mode      : SINGLE")
        process_sequence(
            sequence=args.sequence,
            seq_root=args.seq_root,
            img_w=args.img_w,
            img_h=args.img_h,
            avg_speed=args.avg_speed,
            window=args.window,
            out_path=out_path,
            fatal=True,
        )
        print("═" * 60)
        print("Done.\n")


if __name__ == "__main__":
    main()