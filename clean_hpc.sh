#!/bin/sh

git clean -fd artifacts/
git clean -fd job_out/
git clean -fd checkpoints/
git clean -fd Thesis/
git clean -fd wandb/
git clean -fd trained_models/

rm -r -f wandb/
rm -r -f Thesis/
rm -r -f artifacts/

rm -rf ~/.cache/wandb/