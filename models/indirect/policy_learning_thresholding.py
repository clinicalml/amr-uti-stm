import pandas as pd
import numpy as np

from collections import defaultdict
import itertools
from datetime import datetime

import argparse
import logging
import os
import sys
import json

sys.path.append('../../')
from utils.evaluation_utils import get_iat_broad, get_iat_broad_bootstrapped

from sklearn.metrics import roc_curve



def get_stats_for_setting_combos(preds_df,
                                 resist_prescrip_df,
                                 setting_combos,
                                 num_splits=20,
                                 contra_dict=None,
                                 abx_list=['NIT', 'SXT', 'CIP', 'LVX']):
    stat_columns = [
        'iat_prop', 'broad_prop', 'iat_prop_decision', 'broad_prop_decision',
        'iat_diff_mean', 'iat_diff_std', 'broad_diff_mean', 'broad_diff_std',
        'defer_rate'
    ]

    stats_by_setting = []

    # Combine predicted resistance probabilities with true labels
    preds_prescrip_wide = preds_df.merge(resist_prescrip_df, on='example_id', how='inner')

    setting_combos = list(setting_combos)
    for i, setting in enumerate(setting_combos):
        curr_setting = dict(zip(abx_list, setting))

        # Skip if FNRs for CIP and LVX are not the same
        if curr_setting['CIP']['vme'] != curr_setting['LVX']['vme']: continue

        logging.info(f'Working on combination {i} / {len(setting_combos)}')

        stats_for_curr_setting = defaultdict(list)

        for split in range(num_splits):

            preds_for_split = preds_prescrip_wide[preds_prescrip_wide['split_ct'] == split]

            # Get train/val predictions
            train_preds = preds_for_split[preds_for_split['is_train'] == 1].copy()
            val_preds = preds_for_split[preds_for_split['is_train'] == 0].copy()

            stats_dict_for_split =  get_stats_for_train_val_preds(
                train_preds, val_preds, curr_setting, contra_dict=contra_dict,
                abx_list=abx_list
            )

            doc_iat, doc_broad = get_iat_broad(val_preds, col_name='prescription')

            stats_dict_for_split['iat_diff'] = (stats_dict_for_split['iat_prop'] - doc_iat) * len(val_preds)
            stats_dict_for_split['broad_diff'] = (stats_dict_for_split['broad_prop'] - doc_broad) * len(val_preds)

            for stat in stats_dict_for_split.keys():
                stats_for_curr_setting[stat].append(stats_dict_for_split[stat])

        compiled_stats_for_setting = [curr_setting]

        for stat in stat_columns:
            if stat.endswith('_mean'):
                stat_name = stat[:stat.index('_mean')]
                compiled_stats_for_setting.append(np.mean(stats_for_curr_setting[stat_name]))

            elif stat.endswith('_std'):
                stat_name = stat[:stat.index('_std')]
                compiled_stats_for_setting.append(np.std(stats_for_curr_setting[stat_name]))

            else:
                compiled_stats_for_setting.append(np.mean(stats_for_curr_setting[stat]))

        stats_by_setting.append(compiled_stats_for_setting)

    return convert_dict_to_df(stats_by_setting, stat_columns=stat_columns, abx_list=abx_list)



def get_stats_for_train_val_preds(train_preds, val_preds, curr_setting, contra_dict=None,
                                    bootstrap=False, save_policy=False,
                                    abx_list=['NIT', 'SXT', 'CIP', 'LVX']):

    thresholds = get_thresholds_dict(train_preds, curr_setting, abx_list=abx_list)
    val_policy_df = get_policy_for_preds(val_preds, thresholds,
                                        contra_dict=contra_dict, abx_list=abx_list)

    decision_cohort = val_policy_df[val_policy_df['rec'] != 'defer']

    if bootstrap:
        val_iat, val_broad = get_iat_broad_bootstrapped(val_policy_df, col_name='rec_final')
        val_iat_decision, val_broad_decision = get_iat_broad_bootstrapped(decision_cohort,  col_name='rec_final')
    else:
        val_iat, val_broad = get_iat_broad(val_policy_df, col_name='rec_final')
        val_iat_decision, val_broad_decision = get_iat_broad(decision_cohort,  col_name='rec_final')

    res = {'iat_prop': val_iat,
                'broad_prop': val_broad,
               'iat_prop_decision': val_iat_decision,
               'broad_prop_decision': val_broad_decision,
               'defer_rate': 1-(len(decision_cohort)/len(val_policy_df))}

    if save_policy:
        return res, val_policy_df, thresholds
    else:
        return res


def get_best_combos_for_broads(stats_by_setting,
                               broad_constraints):

    best_val_stats = []

    for broad in broad_constraints:
        best_setting = stats_by_setting[
            stats_by_setting['broad_prop'] < broad
        ].sort_values(by='iat_prop')

        if len(best_setting) > 0:
            best_setting = best_setting.iloc[0:1]
            best_val_stats.append(best_setting)

    return pd.concat(best_val_stats, axis=0)


### Evaluation on test cohort with combos already chosen ###

def get_best_test_outcomes(preds_df, val_outcomes_df,
                          train_resist_df, test_resist_df,
                          broad_constraints, save_policy=False,
                          abx_list=['NIT', 'SXT', 'CIP', 'LVX']):

    all_outcomes = []
    stat_cols = None

    train_preds = preds_df[preds_df['is_train'] == 1]
    test_preds = preds_df[preds_df['is_train'] == 0]

    train_preds_resist_df = train_preds.merge(train_resist_df, on='example_id', how='inner')
    test_preds_resist_df = test_preds.merge(test_resist_df, on='example_id', how='inner')

    if save_policy:
        assert len(broad_constraints) == 1, "Cannot save policy for more than one constraint"

    for max_broad in broad_constraints:

        best_setting = val_outcomes_df[
            val_outcomes_df['broad_prop'] <= max_broad
        ].sort_values(by='iat_prop').iloc[0]

        curr_setting = dict(zip(abx_list, [{'vme': best_setting[abx]} for abx in abx_list]))
        logging.info(curr_setting)

        if save_policy:
            outcomes_for_max_broad, val_policy_df, thresh = get_stats_for_train_val_preds(
                train_preds_resist_df, test_preds_resist_df,
                curr_setting=curr_setting,
                bootstrap=True,
                save_policy=True,
                abx_list=abx_list)

        else:
            outcomes_for_max_broad = get_stats_for_train_val_preds(
                train_preds_resist_df, test_preds_resist_df,
                curr_setting=curr_setting,
                bootstrap=True,
                abx_list=abx_list)

        stat_cols = list(outcomes_for_max_broad.keys()) if stat_cols is None else stat_cols

        all_outcomes.append([max_broad] + [outcomes_for_max_broad[stat] for stat in stat_cols])

    res = pd.DataFrame(all_outcomes, columns=['constraint'] + stat_cols)

    if save_policy:
        return res, val_policy_df, thresh
    else:
        return res

def get_policy_for_constraint(preds_df, val_outcomes_df,
                          train_resist_df, test_resist_df,
                          constraint,
                          abx_list=['NIT', 'SXT', 'CIP', 'LVX']):

    train_preds = preds_df[preds_df['is_train'] == 1]
    test_preds = preds_df[preds_df['is_train'] == 0]

    train_preds_resist_df = train_preds.merge(train_resist_df, on='example_id', how='inner')
    test_preds_resist_df = test_preds.merge(test_resist_df, on='example_id', how='inner')

    best_setting = val_outcomes_df[
        val_outcomes_df['broad_prop'] <= constraint
    ].sort_values(by='iat_prop').iloc[0]

    curr_setting = dict(zip(abx_list, [{'vme': best_setting[abx]} for abx in abx_list]))

    thresholds = get_thresholds_dict(train_preds_resist_df, curr_setting, abx_list=abx_list)
    test_policy_df = get_policy_for_preds(test_preds_resist_df, thresholds,
                                         abx_list=abx_list)

    return test_policy_df



#### Various policy methods given resistance thresholds ####


def get_policy_for_preds(preds_df, thresholds, contra_dict=None,
                        abx_list=['NIT', 'SXT', 'CIP', 'LVX']):

    policy_df = preds_df.copy()

    if contra_dict is not None:
        policy_df['rec'] = policy_df.apply(
            lambda x: get_policy_with_contraindications(x, thresholds, contra_dict,
                abx_list=abx_list), axis=1
        )

    else:
        policy_df['rec'] = policy_df.apply(
            lambda x: get_policy_defer(x, thresholds, abx_list=abx_list), axis=1
        )

    # Fill in deferral with actual antibiotic name
    policy_df['rec_final'] = policy_df.apply(
        lambda x: x['prescription'] if x['rec'] == 'defer' else x['rec'], axis=1
    )

    return policy_df


def get_policy_with_contraindications(row, thresholds, contra_dict,
                                    abx_list=['NIT', 'SXT', 'CIP', 'LVX']):

    for abx in abx_list:
        if row[f'predicted_prob_{abx}'] < thresholds[abx] and row['example_id'] not in contra_dict[abx]:
            return abx

    return "defer"


def get_policy_defer(row, thresholds,
                  abx_list=['NIT', 'SXT', 'CIP', 'LVX']):

    for abx in abx_list:
        if row[f'predicted_prob_{abx}'] < thresholds[abx]:
            return abx

    return "defer"

#### Utility Methods ####

def convert_dict_to_df(stats_by_setting, stat_columns,
                       abx_list=['NIT', 'SXT', 'CIP', 'LVX']):

    data = []

    for setting in stats_by_setting:
        data_for_setting = [setting[0][abx]['vme'] for abx in abx_list] + list(setting[1:])
        data.append(data_for_setting)

    return pd.DataFrame(data, columns=abx_list + stat_columns)


def get_thresholds_dict(preds_df, fnr_setting, abx_list=['NIT', 'SXT', 'CIP', 'LVX']):
    thresholds = {}

    for abx in abx_list:
        threshold, _, _ = get_threshold(preds_df[abx].values,
                                        preds_df[f'predicted_prob_{abx}'].values,
                                        fnr_setting[abx]['vme'])
        thresholds[abx] = threshold

    return thresholds


def get_threshold(is_resistant, resist_probs, fnr):
    desired_tpr = 1 - fnr
    fprs, tprs, thresholds = roc_curve(is_resistant,
                                     resist_probs)

    diffs = [abs(t - desired_tpr) for t in tprs]
    i = diffs.index(min(diffs))
    return thresholds[i], fprs[i], tprs[i]


def create_setting_combos(fnr_vals, abx_list=['NIT', 'SXT', 'CIP', 'LVX']):
    label_settings = defaultdict(list)

    for abx in abx_list:
        label_settings[abx].extend([{'vme': vme} for vme in fnr_vals])

    settings = [label_settings[abx] for abx in abx_list]
    setting_combos = itertools.product(*settings)

    return setting_combos




