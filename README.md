# MCV C6 - Video Analysis Project - Week 3

## Introduction

In Week 3 of the Video Analysis Project we focus on **motion estimation**, **optical flow**, and **object tracking**. Files specific to this Week can be found inside the `Week3/` folder.

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

## Running the off-the-shelf optical flow models

We have implemented **four** different off-the-shelf optical flow models:

- PyFlow
- RAFT (small)
- RAFT (large)
- FlowFormer++

These models can be tested on the KITTI dataset (Sequences 45, image_0, consisting of two frames), using `Week3/optical_flow/main.py`, using the `--model` argument. The script also implements other arguments for further customization, including different visualizations of the results.

While RAFT (small and large) are implemented using PyTorch, PyFlow and FlowFormer++ are implemented directly using their GitHub repositories' codebases, using git submodules to track them.

This is why it is necessary, once the repository is cloned, to execute
```
git submodule update --init --recursive
```
so that the submodules are properly cloned too.

Additionally, there are some particularities to take into account:

### Particularities of executing PyFlow

To compile the files, execute from the root of the repository
```
cd external/pyflow
python setup.py build_ext -i
```

Then, to be able to import them from a Python script:
```
-m pip install -e external/pyflow --no-build-isolation
```

Also, importantly, **if the files are compiled using Windows**, it's necessary to comment line 9 of `external/pyflow/src/project.h`.

### Particularities of executing FlowFormer++

Download `kitti.pth` from [this Google Drive link](https://drive.google.com/drive/folders/1fyPZvcH4SuNCgnBvIJB2PktT5IN9PYPI) and place it at `external/flowformerpp/checkpoints/kitti.pth`. These are the weights of the pre-trained model.

Also, importantly, make sure the Python library `timm` is installed using version 0.4.12 as follows:
```
pip install timm==0.4.12
```


