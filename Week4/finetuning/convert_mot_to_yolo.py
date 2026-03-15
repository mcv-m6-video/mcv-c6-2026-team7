import argparse
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert AI City MOT sequences to YOLO format for multi-camera finetuning"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/AI_CITY_CHALLENGE_2022_TRAIN/train"),
        help="Root directory containing sequences (e.g., .../train/S01/c001)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/yolo_s01s04_trainval_s03_test"),
        help="Output YOLO dataset directory",
    )
    parser.add_argument(
        "--train-seqs",
        nargs="+",
        default=["S01", "S04"],
        help="Sequences used for train/val",
    )
    parser.add_argument(
        "--test-seqs",
        nargs="+",
        default=["S03"],
        help="Sequences used only for test split",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio for train sequences using temporal split",
    )
    parser.add_argument(
        "--image-ext",
        default="jpg",
        choices=["jpg", "png"],
        help="Output image format",
    )
    parser.add_argument(
        "--class-name",
        default="vehicle",
        help="Class name to store in dataset.yaml",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional processing details",
    )
    return parser.parse_args()


def parse_mot_gt(gt_path, frame_width, frame_height):
    annotations = defaultdict(list)
    if not gt_path.exists():
        return annotations

    with gt_path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            fields = line.split(",")
            if len(fields) < 6:
                continue

            frame_idx_1b = int(float(fields[0]))
            left = float(fields[2])
            top = float(fields[3])
            width = float(fields[4])
            height = float(fields[5])

            if width <= 0 or height <= 0:
                continue

            x_center = (left + width / 2.0) / frame_width
            y_center = (top + height / 2.0) / frame_height
            w_norm = width / frame_width
            h_norm = height / frame_height

            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            w_norm = min(max(w_norm, 0.0), 1.0)
            h_norm = min(max(h_norm, 0.0), 1.0)

            # YOLO frame indexing is 0-based in filenames here.
            annotations[frame_idx_1b - 1].append(
                f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

    return annotations


def get_camera_dirs(data_root, sequences):
    camera_dirs = []
    for seq in sequences:
        seq_path = data_root / seq
        if not seq_path.exists():
            continue
        for cam_dir in sorted(seq_path.glob("c*")):
            if cam_dir.is_dir():
                camera_dirs.append((seq, cam_dir.name, cam_dir))
    return camera_dirs


def ensure_output_dirs(output_root):
    for split in ["train", "val", "test"]:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def split_for_frame(frame_idx, total_frames, is_test_sequence, val_ratio):
    if is_test_sequence:
        return "test"

    if total_frames <= 1:
        return "train"

    train_cutoff = int((1.0 - val_ratio) * total_frames)
    train_cutoff = max(1, min(train_cutoff, total_frames - 1))
    return "train" if frame_idx < train_cutoff else "val"


def convert_camera(camera_path, seq_name, cam_name, output_root, val_ratio, is_test_sequence, image_ext, verbose):
    video_path = camera_path / "vdo.avi"
    gt_path = camera_path / "gt" / "gt.txt"

    if not video_path.exists():
        return {"train": 0, "val": 0, "test": 0, "missing_video": True}

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_width <= 0 or frame_height <= 0:
        cap.release()
        return {"train": 0, "val": 0, "test": 0, "invalid_video": True}

    annotations = parse_mot_gt(gt_path, frame_width, frame_height)
    split_counts = {"train": 0, "val": 0, "test": 0}

    desc = f"{seq_name}/{cam_name}"
    for frame_idx in tqdm(range(total_frames), desc=desc):
        ok, frame = cap.read()
        if not ok:
            break

        split = split_for_frame(frame_idx, total_frames, is_test_sequence, val_ratio)
        stem = f"{seq_name.lower()}_{cam_name}_{frame_idx:06d}"

        image_out = output_root / "images" / split / f"{stem}.{image_ext}"
        label_out = output_root / "labels" / split / f"{stem}.txt"

        cv2.imwrite(str(image_out), frame)

        labels = annotations.get(frame_idx, [])
        with label_out.open("w") as handle:
            if labels:
                handle.write("\n".join(labels))

        split_counts[split] += 1

    cap.release()

    if verbose:
        print(
            f"Processed {seq_name}/{cam_name} - "
            f"train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}"
        )

    return split_counts


def write_dataset_yaml(output_root, class_name):
    yaml_path = output_root / "dataset.yaml"
    dataset_text = (
        f"path: {output_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        f"names: ['{class_name}']\n"
    )
    yaml_path.write_text(dataset_text)


def main():
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")

    overlap = set(args.train_seqs).intersection(set(args.test_seqs))
    if overlap:
        raise ValueError(f"Sequences cannot be in both train and test splits: {sorted(overlap)}")

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be between 0 and 1")

    ensure_output_dirs(args.output)

    train_cameras = get_camera_dirs(args.data_root, args.train_seqs)
    test_cameras = get_camera_dirs(args.data_root, args.test_seqs)

    if not train_cameras:
        raise RuntimeError("No train cameras found for --train-seqs")
    if not test_cameras:
        raise RuntimeError("No test cameras found for --test-seqs")

    totals = {"train": 0, "val": 0, "test": 0}
    missing_videos = []
    invalid_videos = []

    for seq_name, cam_name, cam_path in train_cameras:
        counts = convert_camera(
            camera_path=cam_path,
            seq_name=seq_name,
            cam_name=cam_name,
            output_root=args.output,
            val_ratio=args.val_ratio,
            is_test_sequence=False,
            image_ext=args.image_ext,
            verbose=args.verbose,
        )
        if counts.get("missing_video"):
            missing_videos.append(f"{seq_name}/{cam_name}")
            continue
        if counts.get("invalid_video"):
            invalid_videos.append(f"{seq_name}/{cam_name}")
            continue
        totals["train"] += counts["train"]
        totals["val"] += counts["val"]

    for seq_name, cam_name, cam_path in test_cameras:
        counts = convert_camera(
            camera_path=cam_path,
            seq_name=seq_name,
            cam_name=cam_name,
            output_root=args.output,
            val_ratio=args.val_ratio,
            is_test_sequence=True,
            image_ext=args.image_ext,
            verbose=args.verbose,
        )
        if counts.get("missing_video"):
            missing_videos.append(f"{seq_name}/{cam_name}")
            continue
        if counts.get("invalid_video"):
            invalid_videos.append(f"{seq_name}/{cam_name}")
            continue
        totals["test"] += counts["test"]

    write_dataset_yaml(args.output, args.class_name)

    print("\nConversion completed.")
    print(f"Output root: {args.output.resolve()}")
    print(f"Train frames: {totals['train']}")
    print(f"Val frames:   {totals['val']}")
    print(f"Test frames:  {totals['test']}")
    if missing_videos:
        print(f"Missing videos ({len(missing_videos)}): {missing_videos}")
    if invalid_videos:
        print(f"Invalid videos ({len(invalid_videos)}): {invalid_videos}")


if __name__ == "__main__":
    main()