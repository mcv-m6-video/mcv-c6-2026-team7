import os
import math
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────

def parse_homography_file(calib_path: str):
    """
    Lee un calibration.txt con formato AI City:

        Homography matrix: a b c;d e f;g h i
        Reprojection error: X.XX

    """
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(f"calibration.txt no encontrado: {calib_path}")

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
        raise ValueError(f"'Homography matrix' no encontrada en {calib_path}")
    if reproj_error is None:
        reproj_error = float("nan")

    return H, reproj_error


def load_homographies(seq_dir: str, cam_ids: list) -> dict:
    """
    Carga calibration.txt para cada cámara en `cam_ids`.
    """
    homos = {}
    for cam_id in cam_ids:
        cam_name   = f"c{cam_id:03d}"
        calib_path = os.path.join(seq_dir, cam_name, "calibration.txt")
        try:
            H, err = parse_homography_file(calib_path)
            homos[cam_id] = (H, err)
            print(
                f"  [GPS] {cam_name}: homografía cargada "
                f"(reproj_error={err:.3f} px)"
            )
        except FileNotFoundError:
            print(
                f"  [GPS] {cam_name}: calibration.txt no encontrado, "
                "GPS desactivado para esta cámara."
            )
            homos[cam_id] = (None, None)
        except Exception as exc:
            print(f"  [GPS] {cam_name}: error al leer calibración — {exc}")
            homos[cam_id] = (None, None)

    n_valid = sum(1 for v in homos.values() if v[0] is not None)
    print(f"  [GPS] {n_valid}/{len(cam_ids)} cámaras con homografía válida.\n")
    return homos


# ─────────────────────────────────────────────
# Proyección imagen → suelo
# ─────────────────────────────────────────────

def pixel_to_world(H: np.ndarray, px: float, py: float):
    """
    Proyecta un punto imagen (px, py) al plano del suelo usando H.
    """
    pt = H @ np.array([px, py, 1.0], dtype=np.float64)
    return float(pt[0] / pt[2]), float(pt[1] / pt[2])


def _foot_point(row: pd.Series, H: np.ndarray):
    """
    Proyecta el punto central inferior del bbox (pie del objeto) al plano del suelo.
    """
    cx = (row["x1"] + row["x2"]) / 2.0
    cy = row["y2"]
    return pixel_to_world(H, cx, cy)


def tracklet_first_world_pos(tdf: pd.DataFrame, H: np.ndarray):
    """
    Devuelve la posición GPS del PRIMER frame del tracklet
    (ordenado por timestamp, luego por frame_id como desempate).

    Usar el primer/último punto en lugar del centroide medio es clave:
    permite comparar el momento real de entrada/salida de cámara.
    """
    row = tdf.sort_values(["timestamp", "frame_id"]).iloc[0]
    return _foot_point(row, H), float(row["timestamp"])


def tracklet_last_world_pos(tdf: pd.DataFrame, H: np.ndarray):
    """
    Devuelve la posición GPS del ÚLTIMO frame del tracklet
    y su timestamp.
    """
    row = tdf.sort_values(["timestamp", "frame_id"]).iloc[-1]
    return _foot_point(row, H), float(row["timestamp"])


# ─────────────────────────────────────────────
# Spatio-temporal gate  (reemplaza build_spatial_gate)
# ─────────────────────────────────────────────

def build_spatiotemporal_gate(tracklets: list,
                               max_speed_mps: float = 2.0,
                               min_dt_s: float = 0.5) -> np.ndarray:
    """
    Construye una máscara booleana (n × n) donde:
        True  → par permitido para el merge
        False → par bloqueado (movimiento físicamente imposible)
    """
    n = len(tracklets)
    allowed = np.ones((n, n), dtype=bool)

    n_blocked = 0

    for i, ti in enumerate(tracklets):
        first_i = ti.get("world_first")
        last_i  = ti.get("world_last")
        if first_i is None or last_i is None:
            continue
        (wx_first_i, wy_first_i), t_start_i = first_i
        (wx_last_i,  wy_last_i),  t_end_i   = last_i
        if wx_first_i is None:
            continue

        for j, tj in enumerate(tracklets):
            if i >= j:
                continue  # procesar solo triángulo superior; aplicar simétrico al final

            # No aplicar gate a pares de la misma cámara (ya bloqueados por otro mecanismo)
            if ti["cam_id"] == tj["cam_id"]:
                continue

            first_j = tj.get("world_first")
            last_j  = tj.get("world_last")
            if first_j is None or last_j is None:
                continue
            (wx_first_j, wy_first_j), t_start_j = first_j
            (wx_last_j,  wy_last_j),  t_end_j   = last_j
            if wx_first_j is None:
                continue

            # Determinar cuál acaba antes
            if t_end_i <= t_end_j:
                # i acaba antes → i es candidato a "precursor" de j
                dt = t_start_j - t_end_i
                wx_a, wy_a = wx_last_i,  wy_last_i    # último punto de i
                wx_b, wy_b = wx_first_j, wy_first_j   # primer punto de j
            else:
                # j acaba antes → j es candidato a "precursor" de i
                dt = t_start_i - t_end_j
                wx_a, wy_a = wx_last_j,  wy_last_j
                wx_b, wy_b = wx_first_i, wy_first_i

            # Tracklets solapados temporalmente: sin restricción de velocidad
            if dt < min_dt_s:
                continue

            dist  = math.hypot(wx_a - wx_b, wy_a - wy_b)
            speed = dist / dt

            if speed > max_speed_mps:
                allowed[i, j] = False
                allowed[j, i] = False
                n_blocked += 1

    print(f"  [GPS] Spatio-temporal gate: {n_blocked} pares cross-cam bloqueados "
          f"(max_speed={max_speed_mps} u/s, min_dt={min_dt_s} s)")
    return allowed


# ─────────────────────────────────────────────
# Utilidad: cálculo de Xworld/Yworld por fila
# ─────────────────────────────────────────────

def row_to_world(row: pd.Series, H) -> tuple:
    """
    Dado un row de DataFrame y una homografía H (o None),
    devuelve (xw, yw) redondeado a 2 decimales, o (-1, -1) si H es None.
    """
    if H is None:
        return -1, -1
    cx = (row["x1"] + row["x2"]) / 2
    cy = row["y2"]
    xw, yw = pixel_to_world(H, cx, cy)
    return round(xw, 2), round(yw, 2)