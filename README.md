# YOLO Finetuning Pipeline

This repository contains three scripts for the Week4 YOLO workflow:

- `Week4/finetuning/convert_mot_to_yolo.py`: converts AI City MOT data to YOLO format with split `S01+S04 -> train/val` and `S03 -> test`.
- `Week4/finetuning/fine_tune_yolo.py`: trains/finetunes a YOLO model using a dataset YAML.
- `Week4/finetuning/eval_s03_models.py`: evaluates off-the-shelf and/or finetuned models on `S03` test.

## 1) Convert dataset

```bash
source ~/C6/venv/bin/activate
cd /home/pol/C6/mcv-c6-2026-team7
python Week2/finetuning/convert_mot_to_yolo.py \
  --data-root data/AI_CITY_CHALLENGE_2022_TRAIN/train \
  --output data/yolo_s01s04_trainval_s03_test \
  --train-seqs S01 S04 \
  --test-seqs S03 \
  --val-ratio 0.2 \
  --class-name vehicle
```

## 2) Finetune model

```bash
source ~/C6/venv/bin/activate
cd /home/pol/C6/mcv-c6-2026-team7
python Week2/finetuning/fine_tune_yolo.py \
  --data data/yolo_s01s04_trainval_s03_test/dataset.yaml \
  --model /home/pol/C6/mcv-c6-2026-team7/yolo26x.pt \
  --epochs 20 \
  --batch 4 \
  --imgsz 640 \
  --name s01s04_to_s03_yolo26x
```

## 3) Evaluate on S03 test


```bash
source ~/C6/venv/bin/activate
cd /home/pol/C6/mcv-c6-2026-team7
python Week2/finetuning/eval_s03_models.py \
  --data data/yolo_s01s04_trainval_s03_test/dataset.yaml \
  --split test \
  --save-json \
```


