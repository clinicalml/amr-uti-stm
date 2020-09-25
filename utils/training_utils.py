from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier

from sklearn.feature_selection import VarianceThreshold

def apply_variance_threshold(X, selector=None):
    if selector is None:
        selector = VarianceThreshold()
        selector.fit(X)

    X = selector.transform(X)
    return X, selector


def get_base_model(model_class):
    if model_class =='lr':
        clf = LogisticRegression()
    elif model_class == 'dt':
        clf = DecisionTreeClassifier()
    elif model_class == 'rf':
        clf = RandomForestClassifier()
    # elif model_class == 'xgb':
    #     clf = XGBClassifier()
    else:
        raise ValueError("Model class not supported.")

    return clf


def filter_cohort_for_label(cohort_df, resist_df, drug_code):
    eids_with_label = resist_df[~resist_df[drug_code].isna()]['example_id'].values
    return cohort_df[cohort_df['example_id'].isin(eids_with_label)], resist_df[resist_df['example_id'].isin(eids_with_label)]
