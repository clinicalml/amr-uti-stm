from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

import pandas as pd
import numpy as np

ABX_NAME_MAP =  {
    'NIT': 'Nitrofurantoin',
    'SXT': 'TMP-SMX',
    'CIP': 'Ciprofloxacin',
    'LVX': 'Levofloxacin'
}

# Create calibration curve data

def create_calibration_data_df(preds_resist_df, abx_list=['NIT', 'SXT','CIP', 'LVX']):

    full_calibration_df = pd.DataFrame()
   
    for abx in abx_list:
        counts = list(bin_counts(preds_resist_df[f'predicted_prob_{abx}'].values, 10)[:-1])
        mean_probs, true_probs = get_calibration_curve(preds_resist_df, abx)

        counts = counts[:len(mean_probs)]
        calibration_df = pd.DataFrame(list(zip(mean_probs, true_probs, counts)),
                                      columns=['predicted', 'prop_positive', 'count'])

        calibration_df['drug'] = ABX_NAME_MAP[abx]
        calibration_df['bin'] = np.linspace(0.5, 0.5 + len(calibration_df) - 1, len(calibration_df))
        
        full_calibration_df = pd.concat([full_calibration_df, calibration_df], axis=0)

    return full_calibration_df


def get_calibration_curve(preds_df, abx):
                           
    prob_true, prob_pred = calibration_curve(preds_df[abx].values, 
                                             preds_df[f'predicted_prob_{abx}'].values,
                                             n_bins=10)
  
    return list(prob_pred), list(prob_true)

def bin_counts(y_prob, n_bins):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    return np.bincount(binids, minlength=len(bins))


# Create FPR/FNR curve and ROC curve data

def create_fpr_fnr_data(preds_resist_df, abx_list=['NIT', 'SXT','CIP', 'LVX']):
    all_fprs_fnrs_df = pd.DataFrame()

    # Initialize first/last values for FPR / FNR data
    base_fpr_df_1 = pd.DataFrame([[1, 0]], columns=['Threshold', 'value'])
    base_fpr_df_2 = pd.DataFrame([[0, 100]], columns=['Threshold', 'value'])

    base_fnr_df_1 = pd.DataFrame([[1, 100]], columns=['Threshold', 'value'])
    base_fnr_df_2 = pd.DataFrame([[0, 0]], columns=['Threshold', 'value'])

    for abx in abx_list:
        fprs, fnrs, _, thresh = get_roc_curve_data(preds_resist_df, abx=abx)
        
        fpr_df = pd.DataFrame([thresh, 100*np.array(fprs)], index=['Threshold', 'value']).transpose()
        fpr_df = pd.concat([base_fpr_df_1, fpr_df, base_fpr_df_2], axis=0)
        fnr_df = pd.DataFrame([thresh, 100*np.array(fnrs)], index=['Threshold', 'value']).transpose()
        fnr_df = pd.concat([base_fnr_df_1, fnr_df, base_fnr_df_2], axis=0)

        fpr_df['drug'] = ABX_NAME_MAP[abx]
        fpr_df['set'] = 'FPR'
        
        all_fprs_fnrs_df = pd.concat([all_fprs_fnrs_df, fpr_df])
        
        fnr_df['drug'] = ABX_NAME_MAP[abx]
        fnr_df['set'] = 'FNR'
        
        all_fprs_fnrs_df = pd.concat([all_fprs_fnrs_df, fnr_df])

    return all_fprs_fnrs_df


def create_roc_curve_data(preds_resist_df, preds_resist_prev_history_df, abx_list=['NIT', 'SXT','CIP', 'LVX']):

    data_full_cohort, data_prev_history_cohort = {}, {}
    
    for abx in abx_list:
        fprs, _, tprs, _ = get_roc_curve_data(preds_resist_df, abx=abx)
        data_full_cohort[abx] = (fprs, tprs)

        fprs, _, tprs, _ = get_roc_curve_data(preds_resist_prev_history_df, abx=abx)
        data_prev_history_cohort[abx] = (fprs, tprs)

    all_roc_curve_data_df = pd.DataFrame()
   
    for abx, fpr_tprs in data_full_cohort.items():
        fpr_tprs_df = pd.DataFrame([
            fpr_tprs[0],
            fpr_tprs[1]
        ], index=['fpr', 'tpr']).transpose()
        fpr_tprs_df['drug'] = ABX_NAME_MAP[abx]
        fpr_tprs_df['set'] = 'full_cohort'
        
        all_roc_curve_data_df = pd.concat([all_roc_curve_data_df, fpr_tprs_df])
        
    for abx, fpr_tprs in data_prev_history_cohort.items():
        fpr_tprs_df = pd.DataFrame([
            fpr_tprs[0],
            fpr_tprs[1]
        ], index=['fpr', 'tpr']).transpose()
        fpr_tprs_df['drug'] = ABX_NAME_MAP[abx]
        fpr_tprs_df['set'] = 'prior_resist_cohort'
        
        all_roc_curve_data_df = pd.concat([all_roc_curve_data_df, fpr_tprs_df])

    return all_roc_curve_data_df


def get_roc_curve_data(preds_resist_df, abx):
    fprs, tprs, thresh = roc_curve(preds_resist_df[abx].values,
                                  preds_resist_df[f'predicted_prob_{abx}'].values)
    fnrs = [1 - tpr for tpr in tprs]
    
    return fprs, fnrs, tprs, thresh



