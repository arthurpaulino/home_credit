import sys

if len(sys.argv)!=2 :
    print('please call \'python model_validate.py <n_pre_subs>')
    exit()

from features_selection_functions import select_features
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from model_train_function import *
from data_aux_functions import *
import matplotlib.pyplot as plt
from parameters import *
from pickle import load
from time import time
import numpy as np

plt.switch_backend('agg')

train = load(open('intermediary/train.pkl', 'rb'), encoding='latin1').drop(columns=[ID_COL])

selected_features = select_features(heur=CHOSEN_HEURISTIC)

params = LGB_PARAMS(n_rows=train.shape[0]*(N_VALIDATION_SPLITS-1)/N_VALIDATION_SPLITS,
    n_features=len(selected_features),
    mode='submission'
)

train = train[selected_features + [TARGET_COL]]

X = train.drop(columns=[TARGET_COL])
y = train[TARGET_COL]

scores_all_mean = []
scores_all_median = []
n_pre_subs = int(sys.argv[1])

def plot():

    scores_all_adjusted_mean = []
    scores_all_adjusted_median = []

    len_max = len(scores_all_mean[0])

    for scores_all, scores_all_adjusted in zip([scores_all_mean, scores_all_median],
                                               [scores_all_adjusted_mean, scores_all_adjusted_median]):
        for scores_val in scores_all:
            scores_val_adjusted = [score_val for score_val in scores_val]
            for _ in range(len_max - len(scores_val)):
                scores_val_adjusted.append(np.nan)
            scores_all_adjusted.append(scores_val_adjusted)

    data_mean = np.array(scores_all_adjusted_mean)
    data_median = np.array(scores_all_adjusted_median)
    means = np.nanmean(data_mean, axis=0)
    medians = np.nanmean(data_median, axis=0)
    x = range(1, min(len_max, n_pre_subs)+1)

    plt.cla()
    plt.plot(x, means, label='mean blend')
    plt.plot(x, medians, label='median blend', color='r')
    plt.text(x[-1], means[-1], str(means[-1]))
    plt.text(x[-1], medians[-1], str(medians[-1]))
    plt.legend()
    plt.xlabel('# pre-subs')
    plt.ylabel('score')
    plt.title('score vs # pre-subs')
    plt.savefig('output/validation.png')

folds = StratifiedKFold(n_splits=N_VALIDATION_SPLITS, shuffle=True, random_state=int(time()))
for N_VALIDATION_SPLITS_iter, (index_train, index_valid) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[index_train], y.iloc[index_train]
    X_valid, y_valid = X.iloc[index_valid], y.iloc[index_valid]

    print('\n====== validation {}/{} ======='.format(N_VALIDATION_SPLITS_iter+1, N_VALIDATION_SPLITS))
    print('#presubs|\tscore (mean / median)')

    predictions_list = []

    scores_val_mean = []
    scores_val_median = []

    scores_all_mean.append(scores_val_mean)
    scores_all_median.append(scores_val_median)

    for n_pre_subs_iter in range(n_pre_subs):

        print('{}\t|'.format(n_pre_subs_iter+1), end='')

        [predictions] = lgb_train(X_train, y_train, params, [X_valid])

        predictions_list.append(predictions)

        predictions_mean = get_blend(np.array(predictions_list).transpose(), blend_kind='mean')
        predictions_median = get_blend(np.array(predictions_list).transpose(), blend_kind='median')

        score_mean = roc_auc_score(y_valid, predictions_mean)
        score_median = roc_auc_score(y_valid, predictions_median)

        print('{} / {}'.format(score_mean, score_median))

        scores_val_mean.append(score_mean)
        scores_val_median.append(score_median)

        plot()

print('\n=========== report ============')

for N_VALIDATION_SPLITS_iter, scores_val_mean, scores_val_median in zip(range(N_VALIDATION_SPLITS), scores_all_mean, scores_all_median):
    print('\nvalidation {}'.format(N_VALIDATION_SPLITS_iter+1))
    score_max_mean = max(scores_val_mean)
    score_max_median = max(scores_val_median)
    index_mean = scores_val_mean.index(score_max_mean)
    index_median = scores_val_median.index(score_max_median)

    print(' mean blend')
    print('  best score: {}'.format(score_max_mean))
    print('  best # pre-subs: {}'.format(index_mean+1))

    print('\n median blend')
    print('  best score: {}'.format(score_max_median))
    print('  best # pre-subs: {}'.format(index_median+1))

data_mean = np.array(scores_all_mean)
data_median = np.array(scores_all_median)
means_mean = data_mean.mean(axis=0)
means_median = data_median.mean(axis=0)

print('\n-- best mean score --')
index = np.argmax(means_mean)
print(' # pre-subs: {}'.format(index+1))
print(' score: {}'.format(means_mean[index]))

print('\n-- best median score --')
index = np.argmax(means_median)
print(' # pre-subs: {}'.format(index+1))
print(' score: {}\n'.format(means_median[index]))
