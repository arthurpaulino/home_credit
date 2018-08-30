from sklearn.model_selection import StratifiedKFold
from parameters import *
from gc import collect
import lightgbm as lgb
from time import time
import numpy as np

def lgb_train(X, y, params, tests):

    predictions_sums = [np.zeros(test.shape[0]) for test in tests]

    if TRAINING_KIND == 'split':
        folds = StratifiedKFold(n_splits=N_TRAINING_SPLITS, shuffle=True, random_state=int(time()))
        for _, (index_train, index_valid) in enumerate(folds.split(X, y)):
            X_train, y_train = X.iloc[index_train], y.iloc[index_train]
            X_valid, y_valid = X.iloc[index_valid], y.iloc[index_valid]

            dtrain = lgb.Dataset(X_train, y_train)
            dvalid = lgb.Dataset(X_valid, y_valid)

            model = lgb.train(params, dtrain, BOOSTING_ROUNDS, dvalid, early_stopping_rounds=EARLY_STOP, verbose_eval=VERBOSE_EVAL)

            del dtrain, dvalid
            collect()

            for i in range(len(tests)):
                predictions_sums[i] += model.predict(tests[i])/N_TRAINING_SPLITS

            del model
            collect()

        return predictions_sums

    elif TRAINING_KIND == 'all':

        dtrain = lgb.Dataset(X, y)

        params.update({'seed': int(time())})

        model = lgb.train(params, dtrain, BOOSTING_ROUNDS, verbose_eval=VERBOSE_EVAL)

        del dtrain
        collect()

        for i in range(len(tests)):
            predictions_sums[i] += model.predict(tests[i])

        del model
        collect()

        return predictions_sums
