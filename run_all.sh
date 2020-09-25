#!/bin/bash

source setup/paths.sh
conda env list

cd experiments

echo "`date +'%Y-%m-%d %T'`:  Building outcome models - Validation"
bash scripts/validation_scripts/train_outcome_models_validation.sh

echo "`date +'%Y-%m-%d %T'`:  Thresholding experiment - Validation"
bash scripts/validation_scripts/thresholding_validation.sh

echo "`date +'%Y-%m-%d %T'`:  Training models, thresholding on test"
bash scripts/eval_test_scripts/train_outcome_models_eval_test.sh
bash scripts/eval_test_scripts/thresholding_test.sh
