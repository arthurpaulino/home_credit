import sys

if len(sys.argv)!=2:
    print('please call \'python model_gen_presubs.py <n_pre_subs>\'')
    exit()

n_pre_subs = int(sys.argv[1])

from features_selection_functions import select_features
from model_train_function import *
from pickle import load, dump
from os.path import isfile
from parameters import *
from gc import collect
from time import time
import pandas as pd
import numpy as np

train = load(open('intermediary/train.pkl', 'rb'), encoding='latin1')
test_pkl = load(open('intermediary/test.pkl', 'rb'), encoding='latin1')

selected_features = select_features(heur=CHOSEN_HEURISTIC)

params = LGB_PARAMS(n_rows=train.shape[0], n_features=len(selected_features), mode='submission')

train = train[selected_features + [TARGET_COL]]
test = test_pkl[selected_features]

X = train.drop(columns=[TARGET_COL])
y = train[TARGET_COL]

for count in range(n_pre_subs):

    print('iteration {}/{}'.format(count+1, n_pre_subs))

    [predictions] = lgb_train(X, y, params, [test])

    pre_sub = pd.DataFrame()
    pre_sub[ID_COL] = test_pkl[ID_COL]
    pre_sub[TARGET_COL] = predictions

    dump(pre_sub, open('output/pre_sub/{}.pkl'.format(int(time())), 'wb'))
    collect()
