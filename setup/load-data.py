import pandas as pd
import numpy as np
import os
DATA_PATH = os.environ['DATA_PATH']

prescription_df = pd.read_csv(f'{DATA_PATH}/all_prescriptions.csv')
labels_df = pd.read_csv(f'{DATA_PATH}/all_uti_resist_labels.csv')
features_df = pd.read_csv(f'{DATA_PATH}/all_uti_features.csv')

tr_features_df = features_df.query("is_train == 1 & uncomplicated==1").drop(['is_train', 'uncomplicated'], axis=1).sort_values('example_id')
te_features_df = features_df.query("is_train == 0 & uncomplicated==1").drop(['is_train', 'uncomplicated'], axis=1).sort_values('example_id')

tr_features_df.to_csv(f'{DATA_PATH}/train_uncomp_uti_features.csv', index=False)
te_features_df.to_csv(f'{DATA_PATH}/test_uncomp_uti_features.csv', index=False)

resist_data = pd.merge(
    labels_df.query('uncomplicated == 1').drop(['uncomplicated'], axis=1),
    prescription_df[['example_id', 'prescription']],
    on='example_id')

tr_resist_data = resist_data.query('is_train == 1').drop(['is_train'], axis=1).sort_values('example_id')
te_resist_data = resist_data.query('is_train == 0').drop(['is_train'], axis=1).sort_values('example_id')

tr_resist_data.to_csv(f'{DATA_PATH}/train_uncomp_resist_data.csv', index=False)
te_resist_data.to_csv(f'{DATA_PATH}/test_uncomp_resist_data.csv', index=False)

