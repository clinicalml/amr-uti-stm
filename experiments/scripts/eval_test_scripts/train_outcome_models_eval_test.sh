#!/bin/bash

python experiment_train_outcome_models.py --exp_name 'train_outcome_models_eval_test' \
					  --features_path "${DATA_PATH}/train_uncomp_uti_features.csv" \
					  --resist_data_path "${DATA_PATH}/train_uncomp_resist_data.csv" \
					  --test_features_path "${DATA_PATH}/test_uncomp_uti_features.csv" \
					  --test_resist_data_path "${DATA_PATH}/test_uncomp_resist_data.csv" \
                                          --eval_test \
                                          --hyperparams_path "${VAL_OUTCOME_MODEL_PATH}/hyperparameters.json" \
                                          --best_models_path "${VAL_OUTCOME_MODEL_PATH}/best_models.json"
#                                          --hyperparams_path "${REPO_PATH}/models/replication_hyperparameters/hyperparameters.json" \
#                                          --best_models_path "${REPO_PATH}/models/replication_hyperparameters/best_models.json"

