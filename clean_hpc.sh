#!/bin/sh

git clean -fd artifacts/
git clean -fd job_out/
git clean -fd checkpoints/
git clean -fd Thesis/
git clean -fd wandb/
git clean -fd predictions/
git clean -fd trained_models/