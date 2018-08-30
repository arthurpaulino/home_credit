from sklearn.model_selection import StratifiedKFold
from os.path import isfile
from parameters import *
from pickle import load
import lightgbm as lgb
from gc import collect
from time import time
import pandas as pd

train = load(open('intermediary/train.pkl', 'rb'), encoding='latin1')
features = load(open('intermediary/features.pkl', 'rb'), encoding='latin1')

del train[ID_COL]
collect()

X = train[features]
y = train[TARGET_COL]

params = LGB_PARAMS(n_rows=X.shape[0], n_features=X.shape[1], mode='scoring')

while True:

    folds = StratifiedKFold(n_splits=N_TRAINING_SPLITS, shuffle=True, random_state=int(time()))
    for _, (index_train, index_valid) in enumerate(folds.split(X, y)):
        X_train, y_train = X.iloc[index_train], y.iloc[index_train]
        X_valid, y_valid = X.iloc[index_valid], y.iloc[index_valid]

        dtrain = lgb.Dataset(X_train, y_train)
        dvalid = lgb.Dataset(X_valid, y_valid)

        model = lgb.train(params, dtrain, BOOSTING_ROUNDS, dvalid, early_stopping_rounds=EARLY_STOP, verbose_eval=VERBOSE_EVAL)

        del dtrain, dvalid
        collect()

        if not isfile('output/feature_importance.csv'):
            feature_importances = pd.DataFrame()
            feature_importances['feature'] = features
            feature_importances['split'] = 0
            feature_importances['gain'] = 0
            feature_importances['score'] = 0
            feature_importances = feature_importances[['feature', 'split', 'gain', 'score']]
        else:
            feature_importances = pd.read_csv('output/feature_importance.csv')

        for mode in ['split', 'gain']:
            importances = model.feature_importance(importance_type=mode)

            for feature, importance in zip(features, importances):
                feature_importances.loc[feature_importances['feature']==feature, mode] += importance

        feature_importances['score'] = feature_importances['gain']/feature_importances['split']
        feature_importances['score'].fillna(0, inplace=True)

        old_features = list(feature_importances['feature'])

        feature_importances.sort_values(['score', 'feature'], ascending=False, inplace=True)

        new_features = list(feature_importances['feature'])

        count = 0
        for f1, f2 in zip(old_features, new_features):
            if f1!=f2:
                count += 1

        print(count, 'features replaced')

        feature_importances.to_csv('output/feature_importance.csv', index=False)
