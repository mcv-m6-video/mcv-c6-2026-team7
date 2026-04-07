import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv3, YOLO26x, and a finetuned model on the S03 test split"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/yolo_s01s04_trainval_s03_test/dataset.yaml"),
        help="Path to dataset.yaml with S03 mapped to test split",
    )
    parser.add_argument("--split", default="test", help="Dataset split to evaluate")

    parser.add_argument("--yolov3-model", default="yolov3u.pt", help="YOLOv3 model path or hub name")
    parser.add_argument("--yolo26x-model", default="yolo26x.pt", help="YOLO26x model path")
    parser.add_argument(
        "--finetuned-model",
        default="runs/detect/runs/train/s01s04_to_s03_yolo26x_b4/weights/best.pt",
        help="Path to finetuned best.pt",
    )

    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size")
    parser.add_argument("--batch", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--device", default="0", help="CUDA device id (or 'cpu')")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference during evaluation")

    parser.add_argument("--project", default="runs/val", help="Ultralytics project output dir")
    parser.add_argument("--name-prefix", default="s03_compare", help="Prefix for Ultralytics eval run names")
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save COCO-style prediction JSON files from validation",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("runs/val/s03_model_comparison.csv"),
        help="Path to summary CSV file",
    )
    return parser.parse_args()


def _get_metric(results_dict, key_candidates):
    for key in key_candidates:
        if key in results_dict:
            return float(results_dict[key])
    return float("nan")


def evaluate_one(model_name, model_path, args):
    model = YOLO(model_path)
    results = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        half=args.half,
        project=args.project,
        name=f"{args.name_prefix}_{model_name}",
        plots=False,
        save_json=args.save_json,
        verbose=True,
    )

    results_dict = results.results_dict
    precision = _get_metric(results_dict, ["metrics/precision(B)", "metrics/precision"])
    recall = _get_metric(results_dict, ["metrics/recall(B)", "metrics/recall"])
    map50 = _get_metric(results_dict, ["metrics/mAP50(B)", "metrics/mAP50"])
    map50_95 = _get_metric(results_dict, ["metrics/mAP50-95(B)", "metrics/mAP50-95"])
    fitness = _get_metric(results_dict, ["fitness"])

    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "fitness": fitness,
        "save_dir": str(results.save_dir),
    }


def print_summary(rows):
    print("\n=== S03 Evaluation Summary ===")
    print(f"{'Model':<16} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>12} {'Fitness':>10}")
    for row in rows:
        print(
            f"{row['model_name']:<16} "
            f"{row['precision']:>10.4f} "
            f"{row['recall']:>10.4f} "
            f"{row['mAP50']:>10.4f} "
            f"{row['mAP50_95']:>12.4f} "
            f"{row['fitness']:>10.4f}"
        )


def write_csv(rows, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_name", "model_path", "precision", "recall", "mAP50", "mAP50_95", "fitness", "save_dir"]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {args.data}")

    model_specs = [
        ("yolov3", args.yolov3_model),
        ("yolo26x", args.yolo26x_model),
    ]

    rows = []
    failures = []

    for model_name, model_path in model_specs:
        print(f"\nEvaluating {model_name}: {model_path}")
        try:
            row = evaluate_one(model_name, model_path, args)
            rows.append(row)
        except Exception as exc:
            failures.append((model_name, model_path, str(exc)))
            print(f"Failed {model_name}: {exc}")

    if not rows:
        raise RuntimeError("All evaluations failed. Check model paths and environment.")

    print_summary(rows)
    write_csv(rows, args.output_csv)
    print(f"\nSaved summary CSV to: {args.output_csv}")

    if failures:
        print("\nSome evaluations failed:")
        for name, model_path, reason in failures:
            print(f"- {name} ({model_path}): {reason}")


if __name__ == "__main__":
    main()