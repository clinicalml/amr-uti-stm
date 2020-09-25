class HyperparameterGrid():

    def __init__(self):
        DEFAULT_HYPERPARAMETER_GRID = {
            'lr': {
                'C': [0.001, 0.01, 0.1, 1],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear'],
                'intercept_scaling': [1, 1000],
                'max_iter': [1000]
            },
            'dt': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 5, 10, None],
                'min_samples_leaf': [0.01, 0.02, 0.05],
            },
            'rf': {
                'n_estimators': [100, 500, 1000],
                'criterion': ['gini','entropy'],
                'max_depth': [3, 5, 10, None],
                'min_samples_leaf': [0.01, 0.02, 0.05],
            },
            'xgb': {
                'max_depth': [2, 3, 4, 5, 6],
                'eta': [.1, .3, .5],
                'eval_metric': ['auc'],
                'min_child_weight': [1, 3, 5, 7, 9],
                'gamma':[0],
                'scale_pos_weight': [1],
                'bsample': [0.8],
                'n_jobs': [4],
                'n_estimators': [100],
                'colsample_bytree': [0.8],
                'objective': ['binary:logistic'],
            }
        }

        self.param_grids = DEFAULT_HYPERPARAMETER_GRID

