import subprocess
from pathlib import Path

MAIN_PY = Path(__file__).resolve().parent / "main.py"

LSBP_RADII = [48, 32, 16, 8]
T_LOWERS = [2, 3, 5, 7]

BASE_ARGS = [
    "--scale", "0.3333",
    "--save-mask-frames", "false",
    "--method", "lsbp",
    "--morph-kernel-size", "13",
    "--max-aspect-ratio", "3",
    "--merge-distance-threshold", "40",
    "--merge-iou-threshold", "0.4",
]

for radius in LSBP_RADII:
    for t_lower in T_LOWERS:
        print(f"\n{'='*60}")
        print(f"Running: lsbp-radius={radius}, t-lower={t_lower}")
        print(f"{'='*60}")
        cmd = ["python", str(MAIN_PY)] + BASE_ARGS + [
            "--lsbp-radius", str(radius),
            "--t-lower", str(t_lower),
        ]
        subprocess.run(cmd, check=True)