from parameters import *
from gc import collect
import numpy as np

def optimize(df):
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df

def rename_columns(df, suffix):
    for column in df.columns:
        if column not in ['SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU', 'TARGET']:
            df.rename(columns={column:column+suffix}, inplace=True)

def aggr_cols(df, by, past_col, target_cols, degree=1):
    target_cols = list(target_cols)
    for column in ['SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU', past_col]:
        if column in target_cols:
            target_cols.remove(column)
    df = df[by + [past_col] + target_cols].copy()
    w = '__WEIGHTS__'
    df[w] = (-1/(df[past_col]-1))**degree
    for column in target_cols:
        df[column] *= df[w]
    df_groupby = df.groupby(by)
    del df
    collect()
    df_by_sum = df_groupby.sum()
    for column in target_cols:
        df_by_sum[column] /= df_by_sum[w]
    df_by_sum[past_col] = df_groupby.mean()[past_col]

    return df_by_sum.reset_index().drop(columns=[w])

def extract_delta(df, df_orig, by, past_col, target_cols, degree):
    df_deg = aggr_cols(df_orig, by, past_col, target_cols, degree)
    for feature in target_cols:
        feature_name = 'DELTA_'+str(degree)+'_'+feature
        df[feature_name] = (df[feature] - df_deg[feature])/df[feature]
        df[feature_name].fillna(0, inplace=True)

def place_nulls(df):
    for column in df.columns:
        if column=='SK_ID_CURR' or column=='SK_ID_PREV' or column=='SK_ID_BUREAU':
            continue
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            df[column] = df[column].replace(['X', 'XAP', 'XNA'], [np.nan, np.nan, np.nan])

def is_positive(x):
    if x>0:
        return 1
    return 0

def not_negative(x):
    if x>=0:
        return 1
    return 0

def get_cnt_diff(df, df_groupby, feature):
    df['CNT_DIFF_'+feature+'(ENG)'] = df_groupby[feature].transform(lambda x: x.dropna().nunique())

def get_blend(data, blend_kind=BLEND_KIND):
    if blend_kind == 'mean':
        return data.mean(axis=1)
    elif blend_kind == 'median':
        return np.median(data, axis=1)
