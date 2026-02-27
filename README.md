# MCV C6 - Video Analysis Project - Week 2

## Introduction

In Week 2 of the Video Analysis Project we focus on **object detection** and **object tracking**. Files specific to this Week can be found inside the `Week2/` folder.

## Environment Setup

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

This will read the files in `data/` and generate the `parsed_data/` folder with the ready-to-read frames in `.jpg` format. 

