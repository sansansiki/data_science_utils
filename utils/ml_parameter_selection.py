# 使用optuna框架对常见的机器学习分类模型进行超参数调优 Ref:https://optuna.readthedocs.io/zh-cn/latest/index.html
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from imblearn.over_sampling import BorderlineSMOTE
import warnings
import optuna


import optuna
import sklearn.ensemble
from sklearn.metrics import *
import sklearn.model_selection
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from sklearn.metrics import make_scorer
from warnings import simplefilter
simplefilter("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).

class ML_Param_Selection():
    '''
    example:
        mlps = ML_Param_Selection(cv=5,random_state=0,is_unbalance=False,scorer=f1_score)
        study = mlps.fit(X_train,y_train)
        print(study.best_params)

    '''

    def __init__(self, cv=3, random_state=0, is_unbalance=False, scorer=f1_score):
        self.cv = cv
        self.random_state = random_state
        self.is_unbalance = is_unbalance
        self.scorer = scorer

    def xgb_objective(self, trial):
        # 定义参数
        param = {
            # L2 regularization weight, Increasing this value will make model more conservative
            'lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
            # L1 regularization weight, Increasing this value will make model more conservative
            'alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
            # Min loss reduction for further partition on a leaf node. larger,the more conservative
            'gamma': trial.suggest_categorical('gamma', [0, 1, 5]),
            # sampling according to each tree
            'colsample_bytree': trial.suggest_categorical('colsample_bytree',
                                                          [0.6, 0.7, 0.8, 0.9, 1.0]),
            # sampling ratio for training data
            'subsample': trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 0.9, 1.0]),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 1),
            'n_estimators': trial.suggest_int('n_estimators', 10, 500, 5),
            # maximum depth of the tree, signifies complexity of the tree
            'max_depth': trial.suggest_int('max_depth', 3, 25, 1),
            # minimum child weight, larger the term more conservative the tree
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }

        scale_pos_weight = trial.suggest_float(
            'scale_pos_weight', 1, 2) if self.is_unbalance else 1

        # 定义模型
        model = xgb.XGBClassifier(
            **param,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            n_jobs=12
        )

        scorer = make_scorer(self.scorer, pos_label=1)
        score = sklearn.model_selection.cross_val_score(
            model, self.X, self.y, n_jobs=-1, cv=self.cv, scoring=scorer)

        return score.mean()

    def LR_objective(self, trial):
        # 选择模型

        # 定义参数
        params = {
            'tol': trial.suggest_float('tol', 1e-6, 1e-3),
            'C': trial.suggest_float("C", 1e-2, 1),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
        }
        positive_class_weight = trial.suggest_float(
            'class_weight', 0.3, 0.7) if self.is_unbalance else 0.5

        # 定义模型
        model = sklearn.linear_model.LogisticRegression(
            **params,
            random_state=self.random_state,
            max_iter=10000,
            class_weight={0: 1-positive_class_weight,
                          1: positive_class_weight},
            verbose=0)

        # 选择五则交叉下的ROC均值为参数评价
        scorer = make_scorer(self.scorer, pos_label=1)
        score = sklearn.model_selection.cross_val_score(
            model, self.X, self.y, n_jobs=-1, cv=self.cv, scoring=scorer)
        return score.mean()

    def Ridge_objective(self, trial):
        # 选择模型

        # 定义参数
        params = {
            'tol': trial.suggest_float('tol', 1e-6, 1e-3),
            'C': trial.suggest_float("C", 1e-2, 1),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
        }
        positive_class_weight = trial.suggest_float(
            'class_weight', 0.3, 0.7) if self.is_unbalance else 0.5

        # 定义模型
        model = sklearn.linear_model.RidgeClassifier(
            **params,
            class_weight={0: 1-positive_class_weight,
                          1: positive_class_weight},
            random_state=self.random_state)

        # 选择五则交叉下的ROC均值为参数评价
        scorer = make_scorer(self.scorer, pos_label=1)
        score = sklearn.model_selection.cross_val_score(
            model, self.X, self.y, n_jobs=-1, cv=self.cv, scoring=scorer)
        return score.mean()

    def random_forest_obj(self, trial):
        hparams = {
            'max_features': trial.suggest_float('max_features', 0.15, 1.0),
            'max_features': trial.suggest_int('min_samples_split', 2, 14),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 14),
            'max_samples': trial.suggest_float('max_samples', 0.6, 0.99),
            'max_depth': trial.suggest_int('max_depth', 1, 15),
            'n_estimators': trial.suggest_int('n_estimators', 1, 120),
        }
        positive_class_weight = trial.suggest_float(
            'class_weight', 0.3, 0.95) if self.is_unbalance else 0.5

        # 定义模型
        classifier_obj = RandomForestClassifier(**hparams,
                                                class_weight={
                                                    0: 1-positive_class_weight, 1: positive_class_weight},
                                                warm_start=True,
                                                n_jobs=8
                                                )

        # 选择五则交叉下的ROC均值为参数评价
        scorer = make_scorer(self.scorer, pos_label=1)
        score = sklearn.model_selection.cross_val_score(
            classifier_obj, self.X, self.y, n_jobs=-1, cv=cv, scoring=scorer)
        return score.mean()

    def random_forest_obj(self, trial):
        param = {
            # L2 regularization weight, Increasing this value will make model more conservative
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            # sampling ratio for training data
            # 'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
            'subsample': trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 0.9, 1.0]),
            # 'subsample': trial.suggest_categorical('subsample', [0.8,0.9,1.0]),
            'learning_rate': trial.suggest_float('learning_rate',
                                                 0.0001, 1),
            #                [0.008,0.009,0.01,0.012,0.014,0.016,0.018, 0.02,0.05]),
            'n_estimators': trial.suggest_int('n_estimators', 1, 120),
            # maximum depth of the tree, signifies complexity of the tree
            'max_depth': trial.suggest_int('max_depth', 2, 15, 1),
            'n_iter_no_change': trial.suggest_int('n_iter_no_change', 10, 300, 10)
            # minimum child weight, larger the term more conservative the tree
        }

        # 定义模型
        model = GradientBoostingClassifier(
            **param,
            random_state=self.random_state,
            warm_start=True
        )

        # 选择五则交叉下的ROC均值为参数评价
        scorer = make_scorer(self.scorer, pos_label=1)
        score = sklearn.model_selection.cross_val_score(
            model, self.X, self.y, n_jobs=-1, cv=self.cv, scoring=scorer)
        return score.mean()

    def mlp_forest_obj(self, trial):
        param = {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)]),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 1024]),
            'activation': trial.suggest_categorical('activation', ['logistic', 'tanh', 'relu']),
            "solver": trial.suggest_categorical('solver', ['adam']),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 1),
            'n_iter_no_change': trial.suggest_int('n_iter_no_change', 10, 300, 10),
            'alpha': trial.suggest_float('alpha',
                                         0.0001, 1),
            'max_iter': trial.suggest_int('max_iter', 200, 2000, 50),


        }
        # 定义模型
        model = MLPClassifier(
            **param,
            random_state=self.random_state,
            warm_start=True,
            learning_rate='adaptive',
        )

        # 选择五则交叉下的ROC均值为参数评价
        scorer = make_scorer(self.scorer, pos_label=1)
        score = sklearn.model_selection.cross_val_score(
            model, self.X, self.y, n_jobs=-1, cv=self.cv, scoring=scorer)
        return score.mean()

    def fit(self, x, y, n_trials=100):
        self.X = x
        self.y = y

        study = optuna.create_study(direction="maximize")
        study.optimize(self.random_forest_obj, n_trials=n_trials)

        print(study.best_params)
        print(study.best_value)
        return study


simplefilter("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


class Smote_ML_Param_Selection():
    '''经过上采样的特征选择方法'''

    def __init__(self, cv=3, random_state=0, is_unbalance=False, scorer=f1_score):
        self.cv = cv
        self.random_state = random_state
        self.is_unbalance = is_unbalance
        self.scorer = scorer

    def get_params(self, key, trial):
        param = {'XGBoost': {
            # L2 regularization weight, Increasing this value will make model more conservative
            'lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
            # L1 regularization weight, Increasing this value will make model more conservative
            'alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
            # Min loss reduction for further partition on a leaf node. larger,the more conservative
            'gamma': trial.suggest_categorical('gamma', [0, 1, 5]),
            # sampling according to each tree
            'colsample_bytree': trial.suggest_categorical('colsample_bytree',
                                                          [0.6, 0.7, 0.8, 0.9, 1.0]),
            # sampling ratio for training data
            'subsample': trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 0.9, 1.0]),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 1),
            'n_estimators': trial.suggest_int('n_estimators', 10, 500, 5),
            # maximum depth of the tree, signifies complexity of the tree
            'max_depth': trial.suggest_int('max_depth', 3, 25, 1),
            # minimum child weight, larger the term more conservative the tree
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 6)
        }}

        return param[key]

    def get_model(self, model_name, trial):
        param = self.get_params(model_name, trial)
        models = {
            'XGBoost': xgb.XGBClassifier(
                **param,
                objective='binary:logistic',
                n_jobs=12
            )


        }
        return models[model_name]

    def get_objective(self, trial):
        # 定义参数
        model = self.get_model(self.model_name, trial)

        model.fit(self.X_resampled, self.y_resampled)

        # Predict on the test set
        y_pred = model.predict_proba(self.X_test)

        # Calculate AUROC
        auroc = roc_auc_score(self.y_test, y_pred[:, 1])

        return auroc

    def fit(self, X, y, model_name):
        '''划分训练测试集'''

        self.model_name = model_name

        rskf = StratifiedKFold(n_splits=5, shuffle=True,
                               random_state=self.random_state)
        train_index, test_index = next(rskf.split(X, y))
        # for train_index, test_index in rskf.split(X, y):
        X_train, self.X_test = X[train_index], X[test_index]
        y_train, self.y_test = y[train_index], y[test_index]

        sampler = BorderlineSMOTE(random_state=random_state)
        self.X_resampled, self.y_resampled = sampler.fit_resample(
            X_train, y_train)

        object = self.get_objective
        # Create a study object and optimize it
        study = optuna.create_study(direction='maximize')

        study.optimize(object, n_trials=100)

        # Get the best parameters
        best_params = study.best_params
        best_auroc = study.best_value

        print("Best AUROC:", best_auroc)
        print("Best Parameters:", best_params)


ml = smote_ml_Param_Selection()
ml.fit(train_all_x, train_y, 'XGBoost')
