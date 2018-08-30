from features_selection_functions import select_features
from parameters import *
from pickle import load
import lightgbm as lgb
from time import time
import pandas as pd

pd.options.display.precision = 10

train = load(open('intermediary/train.pkl', 'rb'), encoding='latin1').drop(columns=[ID_COL])

selected_features = select_features(heur=CHOSEN_HEURISTIC)

params = LGB_PARAMS(n_rows=train.shape[0], n_features=len(selected_features), mode='submission')

train = train[selected_features + [TARGET_COL]]

X = train.drop(columns=[TARGET_COL])
y = train[TARGET_COL]

dtrain = lgb.Dataset(X, y)

start = time()

eval = lgb.cv(
    params,
    dtrain,
    nfold = N_TRAINING_SPLITS,
    stratified = True,
    num_boost_round = BOOSTING_ROUNDS,
    early_stopping_rounds = EARLY_STOP,
    verbose_eval = CV_VERBOSE_EVAL,
    seed = CV_SEED,
    show_stdv = True
)

end = time()

max_auc_mean_index = eval['auc-mean'].index(max(eval['auc-mean']))
max_auc_mean = eval['auc-mean'][max_auc_mean_index]
max_auc_mean_std = eval['auc-stdv'][max_auc_mean_index]

df = pd.DataFrame({
    'info' : ['mean', '-std', '+std', '1e3std', 'boosts', 'time'],
    'value': [
        max_auc_mean,
        max_auc_mean-max_auc_mean_std,
        max_auc_mean+max_auc_mean_std,
        1000*max_auc_mean_std,
        max_auc_mean_index+1,
        '{:.2f}m'.format((end-start)/60)
    ]
})

print('\n'+df.to_string(index=False))
