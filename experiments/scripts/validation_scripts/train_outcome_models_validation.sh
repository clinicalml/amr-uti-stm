#!/bin/bash

python experiment_train_outcome_models.py --exp_name 'train_outcome_models_validation' \
					  --features_path "${DATA_PATH}/train_uncomp_uti_features.csv" \
					  --resist_data_path "${DATA_PATH}/train_uncomp_resist_data.csv" 
