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

from models.indirect.policy_learning_thresholding import *


parser = argparse.ArgumentParser(description='Process parameters for experiment')

parser.add_argument('--exp_name',
                    type=str, help='Experiment name')

parser.add_argument('--preds_path',
                     type=str, help='Path to predictions file')

parser.add_argument('--resist_data_path',
                    type=str, help='Path to resistance labels CSV for cohort')

parser.add_argument('--contraindications_data_path',
                    required=False, type=str, help='Path to list of EIDs with contraindications')

# Parameters if evaluating on test set
parser.add_argument('--eval_test',
                     action='store_true',
                     help='Flag indicating evaluation on test set')

parser.add_argument('--save_policy',
                     action='store_true',
                     help='Save the policy used on test')

parser.add_argument('--test_resist_data_path',
                    type=str, help='Path to resistance labels CSV for cohort')

parser.add_argument('--val_combo_results_path',
                     type=str,
                     help='Path to performance of different threshold combinations on validation set')


if __name__ == '__main__':
    args = parser.parse_args()
    abx_list = ['NIT', 'SXT', 'CIP', 'LVX']

    log_time = datetime.now().strftime("%d-%m-%Y_%H%M%S")
    log_folder_path = f"experiment_results/thresholding/{args.exp_name}/logs"
    results_path = f"experiment_results/thresholding/{args.exp_name}/results"

    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    logging.basicConfig(filename=os.path.join(log_folder_path, f"tune_thresholds_{log_time}.log"),
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)
    logging.info(args)

    if not args.eval_test:

        setting_combos = create_setting_combos([0.001, 0.015, 0.1, 0.2, 0.3,
                                                0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                                abx_list=abx_list)

        contra_dict = None
        if args.contraindications_data_path:
            contra_eids = pd.read_csv(args.contraindications_data_path)
            contra_dict = {}

            for abx in abx_list:
                contra_eids_for_abx = set(contra_eids[contra_eids['abx'] == abx]["EID"].values)
                contra_dict[abx] = contra_eids_for_abx
                logging.info(f'{len(contra_dict[abx])} EIDs have contraindications to treatment with {abx}.')

        val_stats_by_setting_df = get_stats_for_setting_combos(
            pd.read_csv(args.preds_path),
            pd.read_csv(args.resist_data_path),
            setting_combos,
            abx_list=abx_list,
            contra_dict=contra_dict
        )

        logging.info("Completed calculating stats for all threshold combinations.")

        val_stats_by_setting_df.to_csv(os.path.join(results_path, "val_stats_by_setting.csv"), index=None)

        logging.info("Computing best outcomes for each 2nd line usage constraint...")

        broad_constraints = np.linspace(0.01, 0.50, 50)
        best_outcomes_for_broads = get_best_combos_for_broads(val_stats_by_setting_df,
                                                              broad_constraints)
        best_outcomes_for_broads.to_csv(os.path.join(results_path, "best_val_outcomes_by_max_broad.csv"), index=None)

    else:
        # Choose threshold that was no higher than 10% broad usage on train
        broad_constraints = np.array([0.1])

        preds_df = pd.read_csv(args.preds_path)
        train_resist_df = pd.read_csv(args.resist_data_path)
        test_resist_df = pd.read_csv(args.test_resist_data_path)

        val_outcomes_by_setting_df = pd.read_csv(args.val_combo_results_path)

        # Evaluate this set of thresholds on the test set
        if args.save_policy:
            best_outcomes_for_broads, test_policy_df, thresh = get_best_test_outcomes(
                preds_df, val_outcomes_by_setting_df,
                train_resist_df, test_resist_df,
                broad_constraints,
                save_policy=True,
                abx_list=abx_list)
        else:
            best_outcomes_for_broads = get_best_test_outcomes(
                preds_df, val_outcomes_by_setting_df,
                train_resist_df, test_resist_df,
                broad_constraints,
                abx_list=abx_list)

        best_outcomes_for_broads.to_csv(os.path.join(results_path, "best_test_outcomes_by_max_broad.csv"), index=None)

        if args.save_policy:
            test_policy_df.to_csv(os.path.join(results_path, "test_policy_df.csv"), index=None)
            logging.info(f"Chosen Thresholds: {thresh}")
