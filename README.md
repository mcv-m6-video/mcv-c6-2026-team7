# Week 6 - Video Analysis (C6) - Team 7

**Team members:** Marc Aguilar · Oriol Marín · Pol Rosinés · Biel González · Alejandro Donaire

## Overview

This week we experiment with different **video action spotting** approaches using **SoccerNet** data. The goal is to improve upon a given baseline model.

## Best Approach

Our best-performing model combines:

- Residual Bidirectional GRU
- Temporal Gaussian Label Smoothing

**Model checkpoint:** [Download from Google Drive](https://drive.google.com/file/d/17UPYPjJMmW6grPxKw_5uHFWEyes_Font/view?usp=drive_link)

## Usage

All code is inside the `Week6/` folder. In particular, to reproduce the best model use:

- The main file `main_spotting.py`
- The model file `residual_bigru_TGLS.py`.
- The config file `residual_bigru_TGLS.json`.
