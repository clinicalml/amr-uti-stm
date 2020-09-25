#!/bin/bash

source setup/paths.sh
conda env list

cd experiments

echo "Building outcome models using orignal HPs"
bash scripts/eval_test_scripts/train_outcome_models_eval_test_replication.sh

echo "Done: See notebooks/figures_and_tables.ipynb for next steps"
