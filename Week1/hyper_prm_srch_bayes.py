import subprocess
import time
import re
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# ===============================
# Define Search Space
# ===============================

"""space = [
    Real(0, 20.0, name="alpha"),
    Real(1/3, 1/3, name="scale"),
    Integer(7, 7, name="morph_kernel_size"),
    Real(1000, 1000, name="min_area"),
    Real(5.0, 5.0, name="max_aspect_ratio"),
    Real(0.3, 0.3, name="merge_iou_threshold"),
    Real(100.0, 100.0, name="merge_distance_threshold"),
    Categorical([True], name="adaptive"),
    Real(0.001, 0.1, name="rho"),
    Integer(10, 10, name="num_random_ranks")
]"""

space = [
    Categorical([5], name="alpha"),
    Categorical([1/3], name="scale"),
    Categorical([7], name="morph_kernel_size"),
    Categorical([1000], name="min_area"),
    Categorical([5.0], name="max_aspect_ratio"),
    Categorical([0.3], name="merge_iou_threshold"),
    Real(10, 200, name="merge_distance_threshold"),
    Categorical([True], name="adaptive"),
    Categorical([0.06], name="rho"),
    Categorical([10], name="num_random_ranks")
]

results = []

# ===============================
# Objective Function
# ===============================

@use_named_args(space)
def objective(**params):

    start_time = time.time()

    # Build command
    cmd = [
        "python", "main.py",
        "--save-videos=False",
        "--save-mask-frames=False",
        "--alpha", str(params["alpha"]),
        "--scale", str(params["scale"]),
        "--morph-kernel-size", str(params["morph_kernel_size"]),
        "--min-area", str(params["min_area"]),
        "--max-aspect-ratio", str(params["max_aspect_ratio"]),
        "--merge-iou-threshold", str(params["merge_iou_threshold"]),
        "--merge-distance-threshold", str(params["merge_distance_threshold"]),
        "--num-random-ranks", str(params["num_random_ranks"])
    ]

    if params["adaptive"]:
        cmd.append("--adaptive")
        cmd.extend(["--rho", str(params["rho"])])

    # Run training script
    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    output = process.stdout
    elapsed = time.time() - start_time

    # Extract metrics
    def extract(pattern):
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    mAP = extract(r"mAP@0\.5:\s*([0-9]*\.?[0-9]+)")
    recall = extract(r"Recall:\s*([0-9]*\.?[0-9]+)")
    precision = extract(r"Precision:\s*([0-9]*\.?[0-9]+)")
    f1 = extract(r"F1 Score:\s*([0-9]*\.?[0-9]+)")

    # Store results
    row = {
        **params,
        "mAP@0.5": mAP,
        "Recall": recall,
        "Precision": precision,
        "F1 Score": f1,
        "time": round(elapsed, 2)
    }

    results.append(row)
    pd.DataFrame(results).to_csv(r".\results\bayes_results_MAP_merge_distance_threshold.csv", index=False)

    print("mAP:", mAP)
    #print("F1 Scor:", f1)

    # IMPORTANT: gp_minimize minimizes → return negative mAP
    return -mAP if mAP is not None else 1.0
    #return -f1 if f1 is not None else 1.0


# ===============================
# Run Optimization
# ===============================

res = gp_minimize(
    objective,
    space,
    n_calls=50,            # number of experiments
    n_initial_points=10,   # random warmup
    random_state=42
)

print("\nBest parameters found:")
print(res.x)
print("Best MAP:", -res.fun)
#print("Best F1:", -res.fun)