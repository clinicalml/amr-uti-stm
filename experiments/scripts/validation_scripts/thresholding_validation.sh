#!/bin/bash

python experiment_thresholding.py --exp_name 'thresholding_validation' \
			          --preds_path "${VAL_OUTCOME_MODEL_PATH}/val_predictions.csv" \
				  --resist_data_path "${DATA_PATH}/train_uncomp_resist_data.csv" 
