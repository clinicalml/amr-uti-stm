import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

def get_bootstrapped_auc(test_labels, test_preds, n_samples=1000):
    
    preds_df = pd.DataFrame([test_labels, test_preds],
                            index=['label', 'pred']).transpose()
    
    auc_scores = []
    
    for i in range(n_samples):
        sampled_df = preds_df.sample(len(preds_df), 
                                      replace=True, 
                                      random_state=25+i)
        auc_scores.append(roc_auc_score(sampled_df['label'],
                                        sampled_df['pred']))
    
    z = 1.96
    return np.mean(auc_scores), np.std(auc_scores), (np.mean(auc_scores) - z*np.std(auc_scores), 
                                 np.mean(auc_scores) + z*np.std(auc_scores))


def get_iat_broad_bootstrapped(policy_df, col_name, num_samples=20):
    iats, broads = [], []
    
    for i in range(num_samples):
        policy_df_sampled = policy_df.sample(n=len(policy_df), replace=True,
                                             random_state=10+i)
        
        iat, broad = get_iat_broad(policy_df_sampled, col_name=col_name) 
        iats.append(iat)
        broads.append(broad)
        
    return np.mean(iats), np.mean(broads)
    

def get_iat_broad(policy_df, col_name):
    iat = policy_df.apply(lambda x: x[f'{x[col_name]}']==1, axis=1).mean()
    broad = policy_df.apply(lambda x: x[col_name] in ['CIP', 'LVX'], axis=1).mean() 
    
    return iat, broad


def calculate_ci(proportion, n_samples):
    z = 1.96
    if n_samples == 0: return (0.0, 0.0)
    ci_raw = (proportion - z*np.sqrt((proportion*(1-proportion))/n_samples),
            proportion + z*np.sqrt((proportion*(1-proportion))/n_samples) )
    
    return (round(ci_raw[0], 4), round(ci_raw[1], 4))
