import sys

if len(sys.argv)!=2:
    print('please call \'python model_stack.py <n_pre_subs>\'')
    exit()

n_pre_subs = int(sys.argv[1])

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from model_train_function import *
from pickle import load, dump
from parameters import *
import lightgbm as lgb
from time import time
import pandas as pd
import numpy as np

train = load(open('intermediary/train.pkl', 'rb'), encoding='latin1')
test = load(open('intermediary/test.pkl', 'rb'), encoding='latin1')

feature_importances = pd.read_csv('output/feature_importance.csv')
feature_importances = feature_importances[feature_importances['score']>0]

selected_features = list(feature_importances['feature'])
scores = np.sqrt(feature_importances['score'])
probas = scores/scores.sum()

train = train[selected_features + [TARGET_COL]]

n_features = int(FEATURES_PCT*len(selected_features))

for count in range(n_pre_subs):

    train_1, train_2 = train_test_split(train, test_size=0.5, random_state=int(time()), stratify=train[TARGET_COL])

    stack_1_train, stack_1_test, stack_2_train, stack_2_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for i in range(N_MODELS):

        print('\ntraining model {}'.format(i+1))

        features = list(np.random.choice(selected_features, n_features, replace=False, p=probas))

        for train_1_, train_2_, stack_2_train_, stack_2_test_ in [(train_1, train_2, stack_2_train, stack_2_test),
                                                                  (train_2, train_1, stack_1_train, stack_1_test)]:

            X_1 = train_1_[features]
            y_1 = train_1_[TARGET_COL]

            X_2 = train_2_[features]
            y_2 = train_2_[TARGET_COL]

            params = LGB_PARAMS(n_rows=X_1.shape[0], n_features=X_1.shape[1], mode='submission')

            [stack_2_train_[i], stack_2_test_[i]] = lgb_train(X_1, y_1, params, [X_2, test[features]])

            print(roc_auc_score(y_2, stack_2_train_[i]))

    print('\nensembling models')
    print(roc_auc_score(train_2[TARGET_COL], stack_2_train.mean(axis=1)))
    print(roc_auc_score(train_1[TARGET_COL], stack_1_train.mean(axis=1)))

    print('\nstacking models')

    predictions = pd.DataFrame()

    for stack_train, target, i, stack_test in [(stack_2_train, train_2[TARGET_COL], 2, stack_2_test),
                                               (stack_1_train, train_1[TARGET_COL], 1, stack_1_test)]:

        for stack in [stack_train, stack_test]:
            cols = list(stack.columns)
            stack['mean'] = stack[cols].mean(axis=1)
            stack['median'] = stack[cols].median(axis=1)
            stack['min'] = stack[cols].min(axis=1)
            stack['max'] = stack[cols].max(axis=1)
            stack['std'] = stack[cols].std(axis=1)
            stack['skew'] = stack[cols].skew(axis=1)
            stack['kurt'] = stack[cols].kurt(axis=1)

        dtrain = lgb.Dataset(stack_train, target)

        params = LGB_PARAMS(n_rows=stack_train.shape[0], n_features=stack_train.shape[1], mode='stacking')

        eval = lgb.cv(
            params,
            dtrain,
            nfold = N_TRAINING_SPLITS,
            stratified = True,
            num_boost_round = BOOSTING_ROUNDS,
            early_stopping_rounds = EARLY_STOP,
            verbose_eval = False,
            seed = CV_SEED,
            show_stdv = True
        )

        print(max(eval['auc-mean']))

        [predictions[i]] = lgb_train(stack_train, target, params, [stack_test])

    test[TARGET_COL] = predictions.mean(axis=1)

    dump(test[[ID_COL, TARGET_COL]], open('output/pre_sub/{}.pkl'.format(int(time())), 'wb'))
    collect()
