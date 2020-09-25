import numpy as np
import pandas as pd

import os
import sys
import itertools

sys.path.append("../../")
from models.indirect.train_outcome_models import evaluate_test_cohort
from utils.evaluation_utils import get_iat_broad, calculate_ci


def run_feature_importance_analysis(train_cohort_df, train_resist_df,
                                    test_cohort_df, test_resist_df,
                                    model_class_dict, best_params_dict):

    '''
        Examine predictive performance of resistance models after removing particular sets of features
    '''

    stats = []

    # Get columns to remove in each analysis
    prior_resist_org_cols = [col for col in train_cohort_df.columns if 
                        'micro - prev resist' in col or 'organism' in col]

    prior_exposure_cols = [col for col in train_cohort_df.columns if 'ab class' in col
                           or col.startswith('medication')
                           or 'ab subtype' in col]

    total_abx_cols = [col for col in train_cohort_df.columns if 'ab total' in col]
    colp_cols = [col for col in train_cohort_df.columns if 'colonization' in col]

    feature_set_dropped_cols_map = {
        'Full': [], 
        'Prior antibiotics': prior_exposure_cols, 
        'Prior resistance': prior_resist_org_cols,
        'Colonization pressure': colp_cols,
        'Hospital antibiotic use': total_abx_cols   
    }

    abx_name_map = {
        'NIT': 'Nitrofurantoin',
        'SXT': 'TMP-SMX',
        'CIP': 'Ciprofloxacin',
        'LVX': 'Levofloxacin',
    }

    for feature_set, cols_to_drop in feature_set_dropped_cols_map.items():

        print(f'Feature set to be removed: {feature_set}')
        subcohort_eids = None

        train_features_filtered_df = train_cohort_df.drop(columns=cols_to_drop)
        test_features_filtered_df = test_cohort_df.drop(columns=cols_to_drop)

        auc_dict_for_feature_set, _ , _ = evaluate_test_cohort(train_features_filtered_df, train_resist_df,
                                                              test_features_filtered_df, test_resist_df, 
                                                              best_params_dict, model_class_dict,
                                                              subcohort_eids=subcohort_eids)


        for drug_code, drug_name in abx_name_map.items():
            mean_auc, stdev_auc, ci_auc = auc_dict_for_feature_set[drug_code]

            stats.append(['full', feature_set, drug_name,
                          mean_auc, stdev_auc, ci_auc[0], ci_auc[1]])
                
    stats_df = pd.DataFrame(stats, columns=['cohort', 'model', 'drug',
                                            'AUC', 'STD', 'LCI', 'UCI'])
    stats_df_rounded = stats_df.round(3)

    return stats_df_rounded


def get_prior_exposure_examples(cohort_df):

    # All columns containing features for prior resistance or prior exposure in past 180 days
    exposure_resist_cols = [col for col in cohort_df.columns if ('prev resist' in col
                           or 'ab class' in col
                           or col.startswith('medication')
                           or 'ab subtype' in col) and 'ALL' not in col]

    # Count number of features in this category for each example
    cohort_exposure_resist = cohort_df[exposure_resist_cols].copy()
    prev_resist_exposure_datapoints = pd.concat([cohort_df['example_id'], 
                                           cohort_exposure_resist.sum(axis=1)],
                                          axis=1)
    prev_resist_exposure_history_eids = prev_resist_exposure_datapoints[
        prev_resist_exposure_datapoints[0] > 0
    ]['example_id'].values

    return prev_resist_exposure_history_eids



# Post-hoc analysis of algorithm vs. clinician actions

def get_doc_alg_breakdown(policy_df, col1='rec_final', col2='prescription'):

    # Get columns for algorithm outcomes
    policy_df['alg_iat'] = policy_df.apply(
        lambda x: x[x[col1]]==1, axis=1).astype('int32')

    policy_df['alg_broad'] = policy_df.apply(
        lambda x: x[col1] in ['CIP', 'LVX'], axis=1).astype('int32')
    
    # Get columns for clinician outcomes
    policy_df['doc_iat'] = policy_df.apply(
        lambda x: x[x[col2]]==1, axis=1).astype('int32')

    policy_df['doc_broad'] = policy_df.apply(
        lambda x: x[col2] in ['CIP', 'LVX'], axis=1).astype('int32')
    
    # Count number of specimens in each bucket
    groupby_cols = ['alg_iat', 'alg_broad', 'doc_iat', 'doc_broad']
    policy_group_counts = policy_df.groupby(groupby_cols).count().reset_index()[groupby_cols + ['example_id']]
    
    return policy_group_counts.rename(columns={'example_id': 'count'})


def format_breakdown_df(template_df, new_breakdown_df):
    combos = itertools.product(*[[0,1], [1,0], [0,1], [1,0]])
    new_formatted_df = template_df.copy()
    values = []
    
    for doc_broad, doc_iat, alg_broad, alg_iat in combos:
        relevant_row =  new_breakdown_df[
            (new_breakdown_df['doc_broad'] == doc_broad) & 
            (new_breakdown_df['doc_iat'] == doc_iat) & 
            (new_breakdown_df['alg_broad'] == alg_broad) & 
            (new_breakdown_df['alg_iat'] == alg_iat)
        ]
        if len(relevant_row) == 0:
            values.append(0)
        else:
            values.append(round(relevant_row.iloc[0]['count']))
            
    new_formatted_df['Value'] = values
    return new_formatted_df


# Detailed breakdown of policy outcomes

def compile_all_stats(policy_df):
    stats_data = []
    
    for policy in ['alg', 'doc', 'idsa']:
        for subcohort in ['all', 'decision', 'defer']:
            for antibiotic in ['NIT', 'SXT', 'CIP', 'LVX', 'all', 'first', 'second']:
                n, mean_iat, iat_ci, mean_abx, abx_ci = get_outcomes_for_group(policy_df, policy, antibiotic, subcohort)

                stats_data.append([policy, antibiotic, subcohort,  n,
                                  mean_iat, iat_ci,
                                  mean_abx, abx_ci])

    return pd.DataFrame(stats_data,
                       columns=['policy', 'antibiotic', 'subcohort', 'n',
                                'mean_iat', 'ci_iat',
                                'mean_abx', 'ci_abx'])


def get_outcomes_for_group(policy_df, policy, abx, subcohort):
    
    assert policy in ['doc', 'alg', 'idsa']
    assert abx in ['NIT', 'SXT', 'CIP', 'LVX', 'all', 'first', 'second']
    assert subcohort in ['decision', 'defer', 'all']
    
    # Filter to relevant subcohort
    if subcohort == 'defer':
        policy_df = policy_df[policy_df['alg_raw'] == 'defer']
    elif subcohort == 'decision':
        policy_df = policy_df[policy_df['alg_raw'] != 'defer']
    
    # Filter to relevant policy decision
    if abx in ['NIT', 'SXT', 'CIP', 'LVX']:
        policy_df_relevant = policy_df[policy_df[policy] == abx]
    
    elif abx == 'first':
        policy_df_relevant = policy_df[policy_df[policy].isin(['NIT', 'SXT'])]
    
    elif abx == 'second':
        policy_df_relevant = policy_df[policy_df[policy].isin(['CIP', 'LVX'])]
    
    else:
        policy_df_relevant = policy_df.copy()
        
    # Calculate mean performance
    if len(policy_df_relevant) == 0:
        return 0, 0, 0, 0, 0
    
    mean_prop_iat = policy_df_relevant.apply(lambda x: x[x[policy]] == 1.0, axis=1).mean()
    mean_prop_antibiotic = len(policy_df_relevant)/len(policy_df)
    
    # Calculate confidence intervals
    iat_ci = calculate_ci(mean_prop_iat, len(policy_df_relevant))
    antibiotic_ci = calculate_ci(mean_prop_antibiotic, len(policy_df))

    return len(policy_df_relevant), mean_prop_iat, iat_ci, mean_prop_antibiotic, antibiotic_ci


