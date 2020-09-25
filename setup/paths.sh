#!/bin/bash

# Change this line to reflect where you have stored the data release files
export DATA_PATH="<FULL PATH TO DATA RELEASE FOLDER>"

# Change this line to reflect the location of the repository 
export REPO_PATH="<FULL PATH TO THIS REPOSITORY>"

# DO NOT change these lines; These are assumed elsewhere. 
# Note that while experiment results are stored in the same file tree as the repo, gitignore is set to ignore these results 
export EXP_RESULT_PATH="${REPO_PATH}/experiments/experiment_results"
export TEST_OUTCOME_MODEL_PATH="${EXP_RESULT_PATH}/train_outcome_models/train_outcome_models_eval_test/results"
export TEST_OUTCOME_MODEL_PATH_REP="${EXP_RESULT_PATH}/train_outcome_models/train_outcome_models_eval_test_rep/results"
export VAL_OUTCOME_MODEL_PATH="${EXP_RESULT_PATH}/train_outcome_models/train_outcome_models_validation/results"
export THRESHOLD_RESULT_PATH="${EXP_RESULT_PATH}/thresholding/thresholding_validation/results"

