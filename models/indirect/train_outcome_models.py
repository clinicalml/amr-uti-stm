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
import math

sys.path.append('../../')
from utils.create_cohorts import split_cohort_with_subcohort
from utils.training_utils import apply_variance_threshold, get_base_model, filter_cohort_for_label
from utils.evaluation_utils import get_bootstrapped_auc
from models.indirect.hyperparameter_grids import HyperparameterGrid

from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve


def train_predictive_model_for_model_class(train_cohort_df,
                                           train_resist_df,
                                           train_cohort_info_df,
                                           drug_code, num_trials,
                                           model_class='lr', param_setting=None,
                                           early_stopping_rounds=None,
                                           subcohort_eids=None,
                                           split_by_hosp=False, train_both_hosp=False):

    '''

      Trains a model of type model_class to predict resistance to antibiotic specified by drug_code.

      If param_setting is None, performs a hyperparameter search over grid for specified model class.
      Otherwise, only evaluates model at the provided hyperparameter setting.

      Returns best values of hyperparameters and corresponding validation AUC at that setting.

      Parameters:
        Required:
          - train_cohort_df (Pandas DataFrame): training cohort features
          - train_resist_df (Pandas DataFrame): training cohort resistance labels
          - train_cohort_info_df (Pandas DataFrame): metadata about training cohort. used for generating train/validation splits
          - drug_code (str): Abbreviation for antibiotic for which models are predicting resistance
          - num_trials (int): Number of train/validation spolits to generate
          - model_class (str): Class of predictive model to be trained Must be in ['lr', 'rf', 'dt', 'xgb'].

        Optional:
          - param_setting (dict): Dictionary of hyperparameters corresponding to a single parameter setting.
                                   Used if e.g., you only want to train models at hyperparameters tuned on validation set.

          - early_stopping_rounds (int): Parameter for training XGBoost models. See XGBoost documentation for details.
          - subcohort_eids (list): Parameter for generating train/validation splits.
                                  If provided, each validation set will only contain specimens from within this subcohort.
          - split_by_hosp (bool): Parameter for generating train/validation splits.
                                  If True, will split based on hospital of specimen collection (see next parameter).
          - train_both_hosp (bool): Parameter for generating train/validation splits. Used only when split_by_hosp is True.
                                    If True, will split into a training cohort from MGH/BWH and validation set from BWH.
                                    If False, will split into a training cohort from MGH only and validation set from BWH only.

    '''


    # Set up hyperparameter grid if specific parameter setting is not specified
    if param_setting is None:
        grid = HyperparameterGrid()
        param_grid = grid.param_grids[model_class]
        parameters = list(ParameterGrid(param_grid))

    # If parameter values are specified, only use that setting
    else:
        parameters = [param_setting]

    # Track best hyperparameters and corresponding validation AUC
    best_combo, best_val_auc = None, 0

    for i, param in enumerate(parameters):

        logging.info(f"Training model for combination {i+1} / {len(parameters)} combinations")

        val_aucs, val_rounds = [], []

        for trial in (range(num_trials)):

            # Initialize model of specified class with current hyperparameter setting
            clf = get_base_model(model_class)
            clf.set_params(**param)

            # NOTE: The dataset release does not include meta-data like
            # hospital and patient identifiers, so this cannot be used
            # Split into train / validation cohorts- deterministic for a given cohort and trial number
            if train_cohort_info_df is not None:
                train_train_cohort, train_train_resist_df, val_cohort, val_resist_df = split_cohort_with_subcohort(train_cohort_df,
                                                                                                    train_resist_df,
                                                                                                    train_cohort_info_df,
                                                                                                    seed=24+trial,
                                                                                                    train_prop=.7,
                                                                                                    subcohort_eids=subcohort_eids,
                                                                                                    split_by_hospital=split_by_hosp,
                                                                                                    train_both_hospitals=train_both_hosp)


            else:
                logging.info("No person ID info available. Splitting by example id.")
                train_train_cohort, val_cohort, train_train_resist_df, val_resist_df = train_test_split(train_cohort_df, train_resist_df,
                                                                                                        random_state=24+trial,
                                                                                                        train_size=.7)

            # Filter out any examples that do not have target label for the current antibiotic of interest
            train_train_cohort, train_train_resist_df = filter_cohort_for_label(train_train_cohort, train_train_resist_df, drug_code)
            assert (list(train_train_cohort['example_id'].values) == list(train_train_resist_df['example_id'].values))

            logging.info(f"Train cohort size: {len(train_train_cohort)}")
            logging.info(f"Val cohort size: {len(val_cohort)}")

            # Remove any features with zero variance in the training cohort
            train_x, selector = apply_variance_threshold(train_train_cohort.drop(columns=['example_id']).values)
            val_x, _ = apply_variance_threshold(val_cohort.drop(columns=['example_id']).values,
                                               selector=selector)

            # Fit model to training data, evaluate AUC on validation set
            if model_class == 'xgb':
                clf.fit(train_x,
                        train_train_resist_df[drug_code].values,
                        early_stopping_rounds=early_stopping_rounds,
                        eval_metric='auc', verbose=False,
                        eval_set=[(train_x, train_train_resist_df[drug_code].values),
                                   (val_x, val_resist_df[drug_code].values)])

                if early_stopping_rounds is not None:
                    val_aucs.append(clf.best_score)
                    val_rounds.append(clf.best_iteration)
                else:
                    val_preds = clf.predict_proba(val_x)[:, 1]
                    val_aucs.append(roc_auc_score(val_resist_df[drug_code].values,
                                                  val_preds))
                    val_rounds.append(clf.n_estimators)

            else:
                clf.fit(train_x,
                        train_train_resist_df[drug_code].values)

                val_preds = clf.predict_proba(val_x)[:, 1]
                val_auc = roc_auc_score(val_resist_df[drug_code],
                                        val_preds)
                val_aucs.append(val_auc)

        # Update best hyperparameters / AUC as needed
        if np.mean(val_aucs) > best_val_auc:
            best_combo, best_val_auc = param, np.mean(val_aucs)

            logging.info(f"Best val AUC: {best_val_auc}")
            logging.info(f"Best hyperparameters: {best_combo}")

            if model_class == 'xgb':
                best_combo['n_estimators']  = math.ceil(np.mean(val_rounds)/5)*5

    return best_combo, best_val_auc


def get_best_params_by_model_class(train_cohort_df,
                                   train_resist_df,
                                   train_cohort_info_df,
                                   drug_code,
                                   num_trials=5,
                                   model_classes=['lr', 'dt', 'rf', 'xgb'],
                                   subcohort_eids=None,
                                   split_by_hosp=False,
                                   train_both_hosp=False):

    '''

      Tune hyperparameters for models to predict resistance to antibiotic drug_code
      across several model classes.

      Returns:
        - Dictionary mapping model class to best hyperparameter setting for that model class
        - Dictionary mapping model_class to best validation AUC for that model class

    '''

    hyperparams_by_model, val_aucs_by_model = {}, {}

    for model_class in model_classes:

        logging.info(f'Training models for class {model_class}')
        early_stopping_rounds = 10 if model_class == 'xgb' else None
        best_hyperparams, best_val_auc = train_predictive_model_for_model_class(train_cohort_df,
                                                                               train_resist_df,
                                                                               train_cohort_info_df,
                                                                               drug_code,
                                                                               num_trials,
                                                                               early_stopping_rounds=early_stopping_rounds,
                                                                               model_class=model_class,
                                                                               subcohort_eids=subcohort_eids,
                                                                               split_by_hosp=split_by_hosp,
                                                                               train_both_hosp=train_both_hosp)


        hyperparams_by_model[model_class] = best_hyperparams
        val_aucs_by_model[model_class] = best_val_auc

    return hyperparams_by_model, val_aucs_by_model


def train_models_for_best_params(train_cohort_df,
                               train_resist_df,
                               train_cohort_info_df,
                               drug_code,
                               best_hyperparams,
                               num_trials=20,
                               model_classes=['lr', 'dt', 'rf', 'xgb'],
                               subcohort_eids=None,
                               split_by_hosp=False,
                               train_both_hosp=False):


    '''

      Train models to predict resistance to antibiotic drug_code at the
      best hyperparameter setting chosen from validation set for each model class.

      Returns dictionary mapping model class to validation AUC for that class.

    '''


    val_aucs_by_model = {}

    for model_class in model_classes:

        logging.info(f'Training models for class {model_class}')

        _, best_val_auc = train_predictive_model_for_model_class(train_cohort_df,
                                                               train_resist_df,
                                                               train_cohort_info_df,
                                                               drug_code,
                                                               num_trials,
                                                               model_class=model_class,
                                                               param_setting=best_hyperparams[model_class],
                                                               subcohort_eids=subcohort_eids,
                                                               split_by_hosp=split_by_hosp,
                                                               train_both_hosp=train_both_hosp)

        val_aucs_by_model[model_class] = best_val_auc

    return val_aucs_by_model



def construct_train_val_preds_df(train_cohort_df, train_resist_df,
                                 train_cohort_info_df,
                                 best_hyperparams_dict,
                                 best_models_by_abx,
                                 num_splits=20,
                                 subcohort_eids=None,
                                 split_by_hosp=False,
                                 train_both_hosp=False):


    '''

      Generates predicted probabilities of resistance for multiple train/validataion splits across all antibiotic of interest
      using the optimal tuned hyperparameters (best_hyperparams_dict) and model classes (best_models_by_abx).

      Return pandas DataFrame with columns:

        - example_id: example for which row contains predictions
        - split_ct: number of train/validation split for example in row
        - is_train: binary indicator for whether example in row was in the train or validation part of specified split
        - columns for predicted resistance probability to each antibiotic of interest

    '''


    all_train_val_pred_dfs = []
    abx_list = best_hyperparams_dict.keys()

    for split in range(num_splits):
        if train_cohort_info_df is not None:
            train_train_cohort, train_train_resist_df, val_cohort, val_resist_df = split_cohort_with_subcohort(train_cohort_df,
                                                                                                train_resist_df,
                                                                                                train_cohort_info_df,
                                                                                                seed=24+split,
                                                                                                train_prop=.7,
                                                                                                subcohort_eids=subcohort_eids,
                                                                                                split_by_hospital=split_by_hosp,
                                                                                                train_both_hospitals=train_both_hosp)


        else:
            train_train_cohort, val_cohort, train_train_resist_df, val_resist_df = train_test_split(train_cohort_df, train_resist_df,
                                                                                                    random_state=24+split,
                                                                                                    train_size=.7)

        if subcohort_eids is not None:
            train_uncomp_cohort = train_train_cohort[train_train_cohort['example_id'].isin(subcohort_eids)]
        else:
            train_uncomp_cohort = train_train_cohort.copy()

        train_pred_df = train_uncomp_cohort[['example_id']].copy()
        val_pred_df = val_cohort[['example_id']].copy()

        for abx in abx_list:
            train_train_cohort_for_label, train_train_resist_for_label_df = filter_cohort_for_label(train_train_cohort, train_train_resist_df, abx)
            assert (list(train_train_cohort_for_label['example_id'].values) == list(train_train_resist_for_label_df['example_id'].values))

            train_x, selector = apply_variance_threshold(train_train_cohort_for_label.drop(columns=['example_id']).values)
            train_uncomp_x, _ = apply_variance_threshold(train_uncomp_cohort.drop(columns=['example_id']).values, selector=selector)
            val_x, _ = apply_variance_threshold(val_cohort.drop(columns=['example_id']).values, selector=selector)

            best_model_class = best_models_by_abx[abx]
            clf = get_base_model(best_model_class)
            clf.set_params(**best_hyperparams_dict[abx][best_model_class])
            clf.fit(train_x, train_train_resist_for_label_df[abx].values)

            train_preds = clf.predict_proba(train_uncomp_x)[:, 1]
            val_preds = clf.predict_proba(val_x)[:, 1]

            train_pred_df[f'predicted_prob_{abx}'] = train_preds
            val_pred_df[f'predicted_prob_{abx}'] = val_preds

        train_pred_df['is_train'] = 1
        val_pred_df['is_train'] = 0

        train_val_preds_df = pd.concat([
                train_pred_df, val_pred_df
            ], axis=0)

        train_val_preds_df['split_ct'] = split
        all_train_val_pred_dfs.append(train_val_preds_df)

    return pd.concat(all_train_val_pred_dfs, axis=0)



def evaluate_test_cohort(train_cohort_df, train_resist_df,
                       test_cohort_df, test_resist_df,
                       best_hyperparams_dict, best_models_by_abx,
                       subcohort_eids=None,
                       abx_list=['NIT', 'SXT', 'CIP', 'LVX']):


    '''
      Trains and evaluates models on test cohort for best hyperparameter and model class combination.

      Returns:
        - Dictionary mapping antibiotic to test AUC
        - Dictionary mapping antibiotic to top positive/negative coefficients in predictive model (if model is LR)
        - Pandas DataFrame containing resistance predictions for all examples in training/test cohort
          (similar to the output of construct_train_val_preds_df function)

    '''

    auc_dict, coeffs_dict = {}, {}

    train_preds_df = None
    test_preds_df = test_cohort_df[['example_id']].copy()
    test_preds_df['is_train'] = 0

    for abx in abx_list:

        train_cohort_for_label_df, train_resist_for_label_df = filter_cohort_for_label(train_cohort_df, train_resist_df, abx)
        assert (list(train_cohort_for_label_df['example_id'].values) == list(train_resist_for_label_df['example_id'].values))

        x_train, selector = apply_variance_threshold(train_cohort_for_label_df.drop(columns=['example_id']).values)
        x_test, _ = apply_variance_threshold(test_cohort_df.drop(columns=['example_id']).values,
                                            selector=selector)

        best_model_class = best_models_by_abx[abx]
        clf = get_base_model(best_model_class)
        clf.set_params(**best_hyperparams_dict[abx][best_model_class])
        clf.fit(x_train, train_resist_for_label_df[abx].values)

        # If subcohort is specified (e.g., uncomplicated subcohort examples), we only generate/save training predictions
        # for that subcohort. The model is still trained on all provided examples.
        if subcohort_eids is not None:
            train_cohort_to_predict_df = train_cohort_for_label_df[
                train_cohort_for_label_df['example_id'].isin(subcohort_eids)
            ].sort_values(by='example_id').copy()
        else:
            train_cohort_to_predict_df = train_cohort_for_label_df.copy()

        if train_preds_df is None:
            train_preds_df = train_cohort_to_predict_df[['example_id']].copy()
            train_preds_df['is_train'] = 1

        x_train_to_predict, _ = apply_variance_threshold(train_cohort_to_predict_df.drop(columns=['example_id']).values,
                                                        selector=selector)

        train_preds = clf.predict_proba(x_train_to_predict)[:,1]
        train_preds_df[f'predicted_prob_{abx}'] = train_preds

        # Generate test predictions and compute AUCs
        test_preds = clf.predict_proba(x_test)[:,1]
        test_preds_df[f'predicted_prob_{abx}'] = test_preds

        test_labels = test_resist_df[abx].values
        auc_stats = get_bootstrapped_auc(test_labels, test_preds)

        auc_dict[abx] = auc_stats

        # Extract top coefficients (both positive and negative) from trained model
        if best_model_class == 'lr':
            coeffs_dict[abx] = get_top_coeffs(clf, train_cohort_df.columns)

    all_preds_df = pd.concat([train_preds_df, test_preds_df], axis=0)

    return auc_dict, coeffs_dict, all_preds_df


### Methods for evaluating / analyzing trained models ###


def get_top_coeffs(model, feature_columns, n_coeffs=10):
    top_pos_coeffs_idx = np.argsort(model.coef_[0])[-1*n_coeffs:][::-1]
    top_pos_cols = []

    for coeff_idx in top_pos_coeffs_idx:
        top_pos_cols.append([feature_columns[coeff_idx+1],
                             model.coef_[0][coeff_idx]])

    top_neg_coeffs_idx = np.argsort(model.coef_[0])[:n_coeffs]
    top_neg_cols = []

    for coeff_idx in top_neg_coeffs_idx:
        top_neg_cols.append([feature_columns[coeff_idx+1],
                             model.coef_[0][coeff_idx]])

    colnames = ['feature', 'coeff']
    return pd.DataFrame(top_pos_cols, columns=colnames), pd.DataFrame(top_neg_cols, columns=colnames)


### Utility methods ####

def get_best_models(aucs_by_abx):
    best_model_by_abx = {}

    for abx, aucs_for_abx in aucs_by_abx.items():
        best_auc, best_model_class = 0, None

        for model_class, auc in aucs_for_abx.items():
            if auc > best_auc:
                best_auc, best_model_class = auc, model_class
        best_model_by_abx[abx] = best_model_class

    return best_model_by_abx



