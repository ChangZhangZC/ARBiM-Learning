#!/usr/bin/env bash
# Run from the repository root. Edit path to select the training logs.
path="outputs/il"

tensorboard --logdir "$path"
