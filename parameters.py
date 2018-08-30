from numpy import log

ID_COL = 'SK_ID_CURR'
TARGET_COL = 'TARGET'

####### data processing hyperparameters #######
DEGREE = 1
DEGREE_DELTA = 0

######### feature selection parameters ########
PCT_RANDOM_PICK = 0.5
SCORE_THRESHOLD = 0

CHOSEN_HEURISTIC = 'score_threshold' # check features_selection_functions.py

########### model_validate parameter ##########
N_VALIDATION_SPLITS = 6

######### cross-validation parameters #########
CV_SEED = 42
CV_VERBOSE_EVAL = 10

############### blend parameters ##############
BLEND_KIND = 'mean' # 'mean' or 'median'

############# training parameters #############
TRAINING_KIND = 'split' # 'split' or 'all'. use 'all' if a good number of boosting rounds is known
BOOSTING_ROUNDS = 1000000 # careful here if TRAINING_KIND == 'all'
N_TRAINING_SPLITS = 6 # only used when TRAINING_KIND == 'split'
EARLY_STOP = 200 # only used when TRAINING_KIND == 'split'
VERBOSE_EVAL = False

############# stacking parameters #############
N_MODELS = 20
FEATURES_PCT = 0.6

# http://lightgbm.readthedocs.io/en/latest/Parameters.html
# https://sites.google.com/view/lauraepp/parameters
def LGB_PARAMS(n_rows, n_features, mode):

    MAX_SUBSAMPLE = n_rows

    if TRAINING_KIND=='all':
        subsample_for_bin = MAX_SUBSAMPLE
    elif TRAINING_KIND=='split':
        subsample_for_bin = MAX_SUBSAMPLE*(N_TRAINING_SPLITS-1)/N_TRAINING_SPLITS

    subsample_for_bin = int(subsample_for_bin)

    params = {}

    colsample_bytree = 0.2
    leaves_pct = 0.2

    params['submission'] = dict(
        max_depth = -1,
        num_leaves = int(round(leaves_pct*n_features*colsample_bytree)),
        learning_rate = 0.02,
        max_bin = 2047,
        subsample_for_bin = subsample_for_bin,
        colsample_bytree = colsample_bytree,
        reg_alpha = 1,
        reg_lambda = 512
    )

    params['scoring'] = dict(
        max_depth = -1,
        num_leaves = int(round(leaves_pct*n_features*colsample_bytree)),
        learning_rate = 0.1,
        max_bin = 2047,
        subsample_for_bin = 600,
        colsample_bytree = colsample_bytree,
        reg_alpha = 1,
        reg_lambda = 256
    )

    colsample_bytree = 0.9
    leaves_pct = 0.9

    params['stacking'] = dict(
        max_depth = -1,
        num_leaves = int(round(leaves_pct*n_features*colsample_bytree)),
        learning_rate = 0.01,
        max_bin = 2047,
        subsample_for_bin = subsample_for_bin,
        colsample_bytree = colsample_bytree,
        reg_alpha = 1,
        reg_lambda = 128
    )

    params[mode].update(dict(
        boosting_type = 'gbdt',
        objective = 'binary',
        metric = 'auc',
        verbose = -1,
        device = 'cpu'
    ))

    return params[mode]
