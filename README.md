# MCV C6 - Video Analysis Project

## Setup

1. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Data Setup

The project expects the following directory structure:
```
Root/
├── data/
│   ├── AICity_data
│   └── ai_challenge_s03_c010-full_annotation.xml
├── parsed_data/                              # Frames will be extracted here
└── WeekN/
```

**To extract frames from video** (run once):
```bash
cd ..
python data_processor.py
```

## Run

```bash
cd Week1
python3 main.py
```

This will execute the default background substraction and bounding box detection method. This corresponds to the non-adaptive Gaussian model.

At the end, the configuration used and the metrics obtained will be saved to a unique folder under `results/`. Additionally, three different videos will be saved, showing a variety of results (raw masks, detected bounding boxes, etc) for illustration.

## Method selection and customization

The `main.py` script is highly customizable. Most importantly, using the argument `--method` we can select which of the methods implemented we want to use. Many other hyperparameters can be modified, such as mask post processing parameters or method-specific ones. An example execution with custom parameters looks like this:

```bash
python3 main.py \
    --method mog2 \
    --mog2-history 300 \
    --mog2-var-threshold 25 \
    --mog2-detect-shadows false \
    --save-videos false \
    --save-mask-frames true \
    --scale 0.33
```

The only exception is the implementation of YOLO which has its own `main_yolo.py` file. It does not use any arguments and can be executed directly as follows:

```bash
python3 main_yolo.py
```

## Experiments

To experiment with the different methods we have created separate scripts. For example, `hyper_prm_srch_bayes.py` performs a Bayesian hyperparameter search for our Gaussian models, and `run_lsbp_grid.py` similary performs a grid search for LBSP.



