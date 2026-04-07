# Week 5 - Video Analysis (C6) - Team 7

**Team members:** Marc Aguilar · Oriol Marín · Pol Rosinés · Biel González · Alejandro Donaire

## Overview

This week we experiment with different **video action classification** approaches using **SoccerNet** data. The goal is to improve upon a given baseline model.

## Best Approach

Our best-performing model combines:

- RegNetY 008.
- 2-layer Transformer Encoder with dropout.
- Focal Loss.

**Model checkpoint:** [Download from Google Drive](https://drive.google.com/drive/folders/13tBi4KVAr4-QXlGCRflx7qw-g21UQG1S?usp=drive_link)

## Usage

All code is inside the `Week5/` folder. In particular, to reproduce the best model use:

- The model file `model_classification_transformer.py`.
- The config file `ablation_backbone_focal.json`.