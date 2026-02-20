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
python3 task1.py
```

This will:
1. Extract frames (if not already done)
2. Compute background model using first 25% of frames
3. Generate `bg_mask_output.mp4` with foreground/background segmentation using the remaining 75%.
