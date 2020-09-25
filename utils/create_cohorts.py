import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def split_cohort(cohort_df, resist_df, 
                 cohort_info_df,
                 seed, train_prop=.7):

    '''
        Given a DataFrame containing cohort features and a DataFrame containing resistance
        labels, splits the features / labels data into train / validation sets on the basis
        of person ID. This ensures that there are no individuals with data in both the training
        and validation sets.
    '''
    
    # Split person IDs into train and validation subsets
    pids = sorted(cohort_info_df['person_id'].unique())
    shuffled_pids = shuffle(pids, random_state=seed)
    cutoff = int(len(shuffled_pids)*train_prop)
    train_pids, val_pids = shuffled_pids[:cutoff], shuffled_pids[cutoff:]

    # Extract example IDs corresponding to train / val person IDs
    train_eids = cohort_info_df[cohort_info_df['person_id'].isin(set(train_pids))]['example_id'].values
    val_eids = cohort_info_df[cohort_info_df['person_id'].isin(set(val_pids))]['example_id'].values

    # Extract features for train / val example IDs
    train_cohort_df = cohort_df[cohort_df['example_id'].isin(set(train_eids))]
    val_cohort_df = cohort_df[cohort_df['example_id'].isin(set(val_eids))]

    logging.info(f"Train cohort size: {len(train_cohort_df)}")
    logging.info(f"Validation cohort size: {len(val_cohort_df)}")

    # Extract resistance labels for train / val cohorts - ensure same example ID order by merging
    train_resist_df = train_cohort_df[['example_id']].merge(resist_df, on='example_id', how='inner')
    val_resist_df =  val_cohort_df[['example_id']].merge(resist_df, on='example_id', how='inner')

    assert list(train_cohort_df['example_id'].values) == list(train_resist_df['example_id'].values)
    assert list(val_cohort_df['example_id'].values) == list(val_resist_df['example_id'].values)

    return train_cohort_df, train_resist_df, val_cohort_df, val_resist_df
    

def split_pids(cohort_info_df,
                seed, train_prop=.7): 

    pids = sorted(cohort_info_df['person_id'].unique())
    shuffled_pids = shuffle(pids, random_state=seed)
    cutoff = int(len(shuffled_pids)*train_prop)
    train_pids, val_pids = shuffled_pids[:cutoff], shuffled_pids[cutoff:]
    
    return train_pids, val_pids


def split_pids_by_hospital(cohort_info_df,
                seed, train_both_hospitals=True):

    # Split person IDs into train and validation subsets
    bwh_pids = sorted(cohort_info_df[
        cohort_info_df['hosp'] == 'BWH']['person_id'].unique())
    mgh_pids = sorted(cohort_info_df[
        cohort_info_df['hosp'] == 'MGH']['person_id'].unique())
    
    shuffled_bwh_pids = shuffle(bwh_pids, random_state=seed)    
    cutoff = int(len(shuffled_bwh_pids)*0.5)
    
    # Extract PIDs
    train_bwh_pids, val_bwh_pids = shuffled_bwh_pids[:cutoff], shuffled_bwh_pids[cutoff:]
    val_bwh_pids = list(set(val_bwh_pids).difference(set(mgh_pids)))

    # Extract example IDs corresponding to train / val person IDs
    if train_both_hospitals:
        shuffled_mgh_pids = shuffle(mgh_pids, random_state=seed)   
        train_pids = shuffled_mgh_pids[:-len(train_bwh_pids)] + train_bwh_pids
    
    else:
        train_pids = list(set(mgh_pids).difference(set(bwh_pids)))

    assert(len(set(train_pids).intersection(set(val_bwh_pids))) == 0)
    return train_pids, val_bwh_pids
    

def split_cohort_new(cohort_df, resist_df, 
                     cohort_info_df,
                     seed, train_prop=.7,
                     split_by_hospital=False,
                     train_both_hospitals=True):

    '''
        Given a DataFrame containing cohort features and a DataFrame containing resistance
        labels, splits the features / labels data into train / validation sets on the basis
        of person ID. This ensures that there are no individuals with data in both the training
        and validation sets.
    '''
    if split_by_hospital:
        train_pids, val_pids = split_pids_by_hospital(cohort_info_df, seed, train_both_hospitals) 
    else:
        train_pids, val_pids = split_pids(cohort_info_df, seed, train_prop)

    train_eids = cohort_info_df[cohort_info_df['person_id'].isin(set(train_pids))]['example_id'].values
    val_eids = cohort_info_df[cohort_info_df['person_id'].isin(set(val_pids))]['example_id'].values

    # Extract features for train / val example IDs
    train_cohort_df = cohort_df[cohort_df['example_id'].isin(set(train_eids))]
    val_cohort_df = cohort_df[cohort_df['example_id'].isin(set(val_eids))]

    logging.info(f"Train cohort size: {len(train_cohort_df)}")
    logging.info(f"Validation cohort size: {len(val_cohort_df)}")

    # Extract resistance labels for train / val cohorts - ensure same example ID order by merging

    train_resist_df, val_resist_df = None, None
    if resist_df is not None:
        train_resist_df = train_cohort_df[['example_id']].merge(resist_df, on='example_id', how='inner')
        val_resist_df =  val_cohort_df[['example_id']].merge(resist_df, on='example_id', how='inner')

        assert list(train_cohort_df['example_id'].values) == list(train_resist_df['example_id'].values)
        assert list(val_cohort_df['example_id'].values) == list(val_resist_df['example_id'].values)

    return train_cohort_df, train_resist_df, val_cohort_df, val_resist_df


def split_cohort_with_subcohort(cohort_df, resist_df, cohort_info_df,
                                seed, train_prop, epoch=None, 
                                split_by_hospital=False, train_both_hospitals=False,
                                subcohort_eids=None):

    if subcohort_eids is not None:
        subcohort_df = cohort_df[cohort_df['example_id'].isin(subcohort_eids)].copy()
        non_subcohort_df = cohort_df[~cohort_df['example_id'].isin(subcohort_eids)].copy()

        subcohort_resist_df = resist_df[resist_df['example_id'].isin(subcohort_eids)].copy()
        non_subcohort_resist_df = resist_df[~resist_df['example_id'].isin(subcohort_eids)].copy()

        subcohort_info_df = cohort_info_df[cohort_info_df['example_id'].isin(subcohort_eids)].copy()
        non_subcohort_info_df = cohort_info_df[~cohort_info_df['example_id'].isin(subcohort_eids)].copy()
    
        train_cohort, train_resist_df, val_cohort, val_resist_df = split_cohort_new(subcohort_df,
                                                                                    subcohort_resist_df,
                                                                                    subcohort_info_df,
                                                                                    seed=seed, train_prop=train_prop,
                                                                                    split_by_hospital=split_by_hospital,
                                                                                    train_both_hospitals=train_both_hospitals)
       
        # return train_cohort, train_resist_df , val_cohort, val_resist_df

        # Filter out specimens from training cohort overlapping with individuals in validation cohort
        val_pids = cohort_info_df[cohort_info_df['example_id'].isin(val_cohort['example_id'].values)]['person_id'].values
        non_subcohort_filtered_eids = non_subcohort_info_df[~non_subcohort_info_df['person_id'].isin(val_pids)]['example_id'].values
        
        non_subcohort_filtered_df = non_subcohort_df[non_subcohort_df['example_id'].isin(non_subcohort_filtered_eids)] 
        non_subcohort_filtered_resist_df = non_subcohort_resist_df[non_subcohort_resist_df['example_id'].isin(non_subcohort_filtered_eids)] 
         
        logging.info(f"Number of complicated examples: {len(non_subcohort_filtered_df)}") 

        if epoch is not None:
            num_chunks = int(len(non_subcohort_filtered_df)/5000) + 1
            chunk_idx = epoch % num_chunks
   
            train_cohort = pd.concat([train_cohort, non_subcohort_filtered_df.iloc[(chunk_idx)*5000:(chunk_idx+1)*5000]], axis=0)
            train_resist_df = pd.concat([train_resist_df, non_subcohort_filtered_resist_df.iloc[(chunk_idx)*5000:(chunk_idx+1)*5000]], axis=0)
        else:
            train_cohort = pd.concat([train_cohort, non_subcohort_filtered_df], axis=0)
            train_resist_df = pd.concat([train_resist_df, non_subcohort_filtered_resist_df], axis=0)
        
        assert list(train_cohort["example_id"].values) == list(train_resist_df['example_id'].values)

    else:
        train_cohort, train_resist_df, val_cohort, val_resist_df = split_cohort_new(cohort_df,
                                                                                    resist_df,
                                                                                    cohort_info_df,
                                                                                    seed=seed,
                                                                                    train_prop=train_prop,
                                                                                    split_by_hospital=split_by_hospital,
                                                                                    train_both_hospitals=train_both_hospitals)

    return train_cohort, train_resist_df , val_cohort, val_resist_df



