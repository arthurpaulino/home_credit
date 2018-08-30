from parameters import *
from pickle import load
from os import listdir
import pandas as pd
import numpy as np

def select_features(heur):
    heuristics = dict(
        random_pick = random_pick,
        all_features = all_features,
        score_threshold = score_threshold,
        # ...
    )

    return heuristics[heur]()

################################################################################

def all_features():
    return load(open('intermediary/features.pkl', 'rb'), encoding='latin1')

def random_pick():
    features = all_features()
    return list(np.random.choice(features, size=int(round(PCT_RANDOM_PICK*len(features))), replace=False))

def score_threshold():
    feature_importances = pd.read_csv('output/feature_importance.csv')
    features = list(feature_importances[feature_importances['score']>SCORE_THRESHOLD]['feature'])
    old = feature_importances.shape[0]
    new = len(features)
    print('\nusing {} of {} features ({:.2f}%)\n'.format(new, old, 100*new/old))
    return list(feature_importances[feature_importances['score']>SCORE_THRESHOLD]['feature'])
