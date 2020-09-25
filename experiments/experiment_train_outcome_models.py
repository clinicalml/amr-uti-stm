import pandas as pd
import numpy as np

from collections import defaultdict
from datetime import datetime

import argparse
import logging
import json

import os
import sys
sys.path.append('../')

from models.indirect.train_outcome_models import *


parser = argparse.ArgumentParser(description='Process parameters for experiment')
parser.add_argument('--exp_name',
                     type=str, help='Experiment name')

parser.add_argument('--features_path',
                     type=str, help='Path to features CSV for cohort')

parser.add_argument('--resist_data_path',
                    type=str, help='Path to resistance labels CSV for cohort')

# NOTE: These argument cannot be used in the replication, as it gives a path to
# patient identifiers which are not included in the dataset release
parser.add_argument('--cohort_info_path',
                    type=str, help='Path to cohort metadata CSV for cohort')

parser.add_argument('--subcohort_info_path',
                    type=str, help='Path to subcohort metadata CSV')

# Parameters for running on test set

parser.add_argument('--eval_test',
                     action='store_true',
                     help='Flag indicating evaluation on test set')

parser.add_argument('--test_features_path',
                     type=str, help='Path to features CSV for cohort')

parser.add_argument('--test_resist_data_path',
                    type=str, help='Path to resistance labels CSV for cohort')

# Tuned hyperparameters from validation runs

parser.add_argument('--hyperparams_path',
                    type=str, help='Path to tuned hyperparameters on validation set')

parser.add_argument('--best_models_path',
                    type=str, help='Path to best model classes for each target tuned on validation set')

# Additional splitting parameters

# NOTE: These argument cannot be used in the replication, as they require
# knowledge of locations which are not included in the dataset release
parser.add_argument('--split_by_hosp',
                    action='store_true',
                    help='Perform train / validation splitting by hospital (MGH/BWH)')

parser.add_argument('--train_both_hosp',
                    action='store_true',
                     help='Used if splitting train/val by hospital. If true, use data from both hospitals in training')


if __name__ == '__main__':

    args = parser.parse_args()

    abx_list = ['NIT', 'SXT', 'CIP', 'LVX']
    hyperparams_by_abx, val_aucs_by_abx = {}, {}

    # Create file for logging output
    log_time = datetime.now().strftime("%d-%m-%Y_%H%M%S")
    log_folder_path = f"experiment_results/train_outcome_models/{args.exp_name}/logs"
    results_path = f"experiment_results/train_outcome_models/{args.exp_name}/results"

    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    log_time = datetime.now().strftime("%d-%m-%Y_%H%M%S")

    logging.basicConfig(filename=f"experiment_results/train_outcome_models/{args.exp_name}/logs/train_predictive_models_{log_time}.log",
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)
    logging.info(args)
    logging.info("Reading in data...")

    # Load in data for training models
    train_cohort_df = pd.read_csv(args.features_path)
    train_resist_df = pd.read_csv(args.resist_data_path)

    train_cohort_info_df=None
    if args.cohort_info_path is not None:
        train_cohort_info_df = pd.read_csv(args.cohort_info_path)

    subcohort_eids = None
    if args.subcohort_info_path is not None:
        subcohort_eids=pd.read_csv(args.subcohort_info_path)['example_id'].values
        logging.info(f"{len(subcohort_eids)} examples in specified subcohort.")


    if not args.eval_test:

        for abx in abx_list:
            logging.info(f'Training models for {abx}')
            best_params_for_abx, val_aucs_for_abx = get_best_params_by_model_class(train_cohort_df,
                                                                                   train_resist_df,
                                                                                   train_cohort_info_df=train_cohort_info_df,
                                                                                   drug_code=abx,
                                                                                   model_classes=['lr'], #, 'rf'],
                                                                                   subcohort_eids=subcohort_eids,
                                                                                   split_by_hosp=args.split_by_hosp,
                                                                                   train_both_hosp=args.train_both_hosp)

            logging.info(f'Evaluating tuned models for {abx}')
            val_aucs_for_abx_best_params = train_models_for_best_params(train_cohort_df,
                                                                   train_resist_df,
                                                                   train_cohort_info_df,
                                                                   drug_code=abx,
                                                                   model_classes=['lr'], #, 'rf'],
                                                                   best_hyperparams=best_params_for_abx,
                                                                   subcohort_eids=subcohort_eids,
                                                                   split_by_hosp=args.split_by_hosp,
                                                                   train_both_hosp=args.train_both_hosp)


            hyperparams_by_abx[abx] = best_params_for_abx
            val_aucs_by_abx[abx] = val_aucs_for_abx_best_params

        with open(os.path.join(results_path, 'hyperparameters.json'), 'w') as fp:
            json.dump(hyperparams_by_abx, fp)

        with open(os.path.join(results_path, 'val_aucs.json'), 'w') as fp:
            json.dump(val_aucs_by_abx, fp)

        logging.info("Finding best model based on validation AUC")
        best_models_by_abx = get_best_models(val_aucs_by_abx)

        with open(os.path.join(results_path, 'best_models.json'), 'w') as fp:
            json.dump(best_models_by_abx, fp)

        logging.info("Construction train / validation predictions to be saved")
        train_val_preds_df = construct_train_val_preds_df(train_cohort_df, train_resist_df,
                                                         train_cohort_info_df,
                                                         hyperparams_by_abx,
                                                         best_models_by_abx,
                                                         subcohort_eids=subcohort_eids,
                                                         split_by_hosp=args.split_by_hosp,
                                                         train_both_hosp=args.train_both_hosp)
        train_val_preds_df.to_csv(os.path.join(results_path, 'val_predictions.csv'), index=None)


    else:
        with open(args.hyperparams_path) as f:
            hyperparams_by_abx = json.load(f)

        with open(args.best_models_path) as f:
            best_models_by_abx = json.load(f)

        test_cohort_df = pd.read_csv(args.test_features_path)
        test_resist_df = pd.read_csv(args.test_resist_data_path)

        auc_dict, top_coeffs_dict, full_preds_df = evaluate_test_cohort(train_cohort_df, train_resist_df,
                                                                       test_cohort_df, test_resist_df,
                                                                       best_hyperparams_dict=hyperparams_by_abx,
                                                                       best_models_by_abx=best_models_by_abx,
                                                                       subcohort_eids=subcohort_eids,
                                                                       abx_list=abx_list)

        for abx, (top_pos_coeffs, top_neg_coeffs) in top_coeffs_dict.items():
            top_pos_coeffs.to_csv(os.path.join(results_path, f'pos_coeffs_{abx}.csv'), index=None)
            top_neg_coeffs.to_csv(os.path.join(results_path, f'neg_coeffs_{abx}.csv'), index=None)

        with open(os.path.join(results_path, 'test_aucs.json'), 'w') as fp:
            json.dump(auc_dict, fp)

        full_preds_df.to_csv(os.path.join(results_path, 'test_predictions.csv'), index=None)




