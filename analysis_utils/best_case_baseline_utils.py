import pandas as pd
import os
import sys

sys.path.append('../../')
from models.indirect.policy_learning_thresholding import get_iat_broad
from utils.evaluation_utils import calculate_ci

def get_best_case_idsa_baseline(resist_df,
                               switch_props,
                               option='doc',
                               avoid_nit_eids=None):

    '''
      Implementation of "best-case" modification of IDSA practice guidelines.
    '''
    
    assert option in ['total_rand', 'doc']

    all_iat_broad_stats = []
    resist_df = resist_df.copy()
    
    for p in switch_props:

        resist_df['idsa'] = resist_df.apply(lambda x: 'CIP' if x['example_id']
                                           in set(avoid_nit_eids) else 'NIT', axis=1)
        if option == 'total_rand':
            cip_cohort = resist_df[resist_df['idsa'] == 'CIP']
            prop_cip, iat_cip = len(cip_cohort) / len(resist_df), cip_cohort['CIP'].mean()
            
            nit_cohort = resist_df[resist_df['idsa'] == 'NIT']
            iat_nit_cohort, iat_cip_switch = nit_cohort['NIT'].mean(), nit_cohort['CIP'].mean()
            
            iat = prop_cip * iat_cip + (1-prop_cip) * (p*iat_cip_switch + (1-p)*iat_nit_cohort)
            broad = prop_cip + p*(1-prop_cip)

        elif option == 'doc':
            fixed_cohort = resist_df[~((resist_df['idsa'] == 'NIT') & 
                                       resist_df['prescription'].isin(['CIP', 'LVX']))]
            prop_fixed = len(fixed_cohort) / len(resist_df)
            iat_fixed, broad_fixed = get_iat_broad(fixed_cohort, col_name='idsa')
            switch_cohort = resist_df[(resist_df['idsa'] == 'NIT') & 
                                       resist_df['prescription'].isin(['CIP', 'LVX'])]
            
            iat_idsa_switch = switch_cohort['NIT'].mean()
            iat_doc_switch, _ = get_iat_broad(switch_cohort, col_name='prescription')
            
            iat = prop_fixed*iat_fixed + (1-prop_fixed)*(p*iat_doc_switch + (1-p)*iat_idsa_switch)
            broad = prop_fixed*broad_fixed + p*(1-prop_fixed)
            
        all_iat_broad_stats.append([iat, broad])
        
    return pd.DataFrame(all_iat_broad_stats, columns=['iat', 'broad'])


def get_best_case_stats(policy_df, p=.18):
    fixed_cohort = policy_df[~((policy_df['idsa'] == 'NIT') & 
                               policy_df['doc'].isin(['CIP', 'LVX']))]
    prop_fixed = len(fixed_cohort) / len(policy_df)
    iat_fixed, broad_fixed = get_iat_broad(fixed_cohort, col_name='idsa')

    switch_cohort = policy_df[(policy_df['idsa'] == 'NIT') & 
                               policy_df['doc'].isin(['CIP', 'LVX'])]
    
    # Antibiotic distribution stats
    
    prop_CIP = switch_cohort['doc'].value_counts(normalize=True).get('CIP')
    prop_LVX = 1-prop_CIP
    
    prop_CIP_all = prop_fixed*broad_fixed + p*(1-prop_fixed)*prop_CIP
    prop_LVX_all = p*(1-prop_fixed)*prop_LVX
    
    prop_CIP_ci, prop_LVX_ci = calculate_ci(prop_CIP_all, len(policy_df)), calculate_ci(prop_LVX_all, len(policy_df))

    
    broad = prop_fixed*broad_fixed + p*(1-prop_fixed)
    narrow = 1 - broad
    narrow_ci, broad_ci =  calculate_ci(narrow, len(policy_df)), calculate_ci(broad, len(policy_df))
    

    # IAT stats
    
    iat_idsa_switch = switch_cohort['NIT'].mean()
    iat_doc_switch, _ = get_iat_broad(switch_cohort, col_name='doc')

    iat = prop_fixed*iat_fixed + (1-prop_fixed)*(p*iat_doc_switch + (1-p)*iat_idsa_switch)
    iat_all_ci = calculate_ci(iat, len(policy_df))
    
    all_NIT_size = (len(fixed_cohort[fixed_cohort['idsa'] == 'NIT'])) + (1-p)*len(switch_cohort)    
    iat_fixed_NIT = fixed_cohort[fixed_cohort['idsa'] == 'NIT']['NIT'].mean()
    iat_NIT = len(fixed_cohort[fixed_cohort['idsa'] == 'NIT']) / all_NIT_size  * iat_fixed + (1-p)*len(switch_cohort) / all_NIT_size * iat_idsa_switch
    iat_NIT_ci = calculate_ci(iat_NIT, all_NIT_size)
    
    iat_fixed_broad = fixed_cohort[fixed_cohort['idsa'] == 'CIP']['CIP'].mean()
    all_broad_size = len(fixed_cohort[fixed_cohort['idsa'] == 'CIP']) + p*len(switch_cohort)
    iat_broad = len(fixed_cohort[fixed_cohort['idsa'] == 'CIP']) / all_broad_size * iat_fixed_broad + p*len(switch_cohort)/all_broad_size * iat_doc_switch
    iat_broad_ci = calculate_ci(iat_broad, all_broad_size)
    
    all_CIP_size = len(fixed_cohort[fixed_cohort['idsa'] == 'CIP']) + p*len(switch_cohort)*prop_CIP
    iat_doc_CIP = switch_cohort[switch_cohort['doc'] == 'CIP']['CIP'].mean()
    iat_CIP = len(fixed_cohort[fixed_cohort['idsa'] == 'CIP']) / all_CIP_size * iat_fixed_broad + p*prop_CIP*len(switch_cohort)/all_CIP_size * iat_doc_CIP
    iat_CIP_ci = calculate_ci(iat_CIP, all_CIP_size)
    
    iat_doc_LVX = switch_cohort[switch_cohort['doc'] == 'LVX']['LVX'].mean()
    
    #print(len(switch_cohort[switch_cohort['doc'] == 'LVX']))
    #print(switch_cohort[switch_cohort['doc'] == 'LVX']['LVX'].sum())
    iat_LVX = iat_doc_LVX
    iat_LVX_ci = calculate_ci(iat_LVX, len(switch_cohort[switch_cohort['doc'] == 'LVX'])*p)
    
    abx_prop_stats = {
        'all': (broad, broad_ci),
        'narrow': (narrow, narrow_ci),
        'NIT': (narrow, narrow_ci),
        'CIP': (prop_CIP_all, prop_CIP_ci),
        'LVX': (prop_LVX_all, prop_LVX_ci)
    }
    
    iat_stats = {
        'all': (iat, iat_all_ci),
        'NIT': (iat_NIT, iat_NIT_ci),
        'broad': (iat_broad, iat_broad_ci),
        'CIP': (iat_CIP, iat_CIP_ci),
        'LVX': (iat_LVX, iat_LVX_ci),
    }
    
    return abx_prop_stats, iat_stats
