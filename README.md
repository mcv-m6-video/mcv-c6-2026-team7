# Week 7 - Video Analysis (C6) - Team 7

**Team members:** Marc Aguilar · Oriol Marín · Pol Rosinés · Biel González · Alejandro Donaire

## Slides
The presentation slides are available in [Google Slides](https://docs.google.com/presentation/d/1dX51hq7ZbdBtsOfERDI_usZvNYEfWUeEeKxJrPh7XDU/edit?usp=sharing) and as a [PDF file](Project%202%20presentation%20-%20Team%207.pdf).

## Overview

This week we experiment with different **video action spotting** approaches using **SoccerNet** data. The goal is to improve upon a given baseline model. In particular, this week we are explicitly asked to perform temporal aggregation at some point in the architecture.

## Best Approach

Our best-performing model combines:

- X3D Backbone
- Temporal U-Net, with modifications:
    - Dropout
    - Dilated convolutions
    - Squeeze and excitation blocks
- Temporal Gaussian Label Smoothing

**Model checkpoint:** [Download from Google Drive](https://drive.google.com/file/d/1348nX-FxoGLIIWT7PL80JxIQ5gflc5sL/view?usp=drive_link)

## Usage

All code is inside the `Week7/` folder. In particular, to reproduce the best model use:

- The main file `main_spotting.py`
- The model file `unet_ablation.py`.
- The config file `config_final.json`.
