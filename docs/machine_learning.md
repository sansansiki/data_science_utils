# Utils for Machine Learning

## 1. Parameter_selection

- [source code](../utils/ml_parameter_selection.py)
- Description: This util is used to select the best parameters for a given machine learning model. It uses OPTUNA(https://optuna.readthedocs.io/zh-cn/latest/index.html) to find the best parameters and then trains the model with the best parameters. 

- class ML_Param_Selection() 
  - Description: This class is used to select the best parameters for a given machine learning model. It uses OPTUNA to find the best parameters and then trains the model with the best parameters. 

  - __init__(self, cv=3, random_state=0, is_unbalance=False, scorer=f1_score) 

    - Description: This is the constructor of the class. It takes in the model, X_train, y_train, X_test, y_test, n_trials, and n_jobs as parameters. 

    - Parameters: cv: Cross-validation strategy. random_state: Random state for reproducibility. is_unbalance: Whether the dataset is unbalanced. scorer: Select scoring function. 

  - fit(self, x, y, n_trials=100) 
    - Return: optuna.study  The Result of the best param    
- class Smote_ML_Param_Selection() 
  - Description: This class is used to select the best parameters for a given machine learning model. It uses OPTUNA to find the best parameters and then trains the model with the best parameters, and use smote method to solove unbalanced data.

  - __init__(self, random_state=0, is_unbalance=False, scorer=f1_score) 

    - Description: This is the constructor of the class. It takes in the model, X_train, y_train, X_test, y_test, n_trials, and n_jobs as parameters. 

    - Parameters: cv: Cross-validation strategy. random_state: Random state for reproducibility. is_unbalance: Whether the dataset is unbalanced. scorer: Select scoring function. 

  - fit(self, x, y, n_trials=100) 
    - Return: optuna.study  The Result of the best param
    - example :
      `ml = smote_ml_Param_Selection()
       ml.fit(train_all_x,train_y,'XGBoost')`


## 2. Random Seed Set

- [source code](../utils/machine_learning.py.py)
- Description: This util is used to set the random seed for a given machine learning model. It takes in the model, X_train, y_train, X_test, y_test, n_trials, and n_jobs as parameters. 

- setup_seed(seed)
