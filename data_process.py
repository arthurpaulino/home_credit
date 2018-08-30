from data_aux_functions import *
from parameters import *
from pickle import dump
from gc import collect
from time import time
import pandas as pd
import sys

def eng_bureau(df):

    df['MONTHS_BALANCE'] = df['DAYS_CREDIT']//30

    df['AMT_CREDIT_SUM_DEBT'].fillna(0, inplace=True)

    for feature in ['AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_OVERDUE']:
        df[feature+'_PER_REMAINING_DAY(ENG)'] = df['DAYS_CREDIT_ENDDATE'].apply(not_negative)*df[feature]/(1+df['DAYS_CREDIT_ENDDATE'])

    df['AMT_PAYMENT(ENG)'] = df['AMT_CREDIT_SUM'] - df['AMT_CREDIT_SUM_DEBT']
    df['ENDDATE_DAYDIFF(ENG)'] = df['DAYS_ENDDATE_FACT'] - df['DAYS_CREDIT_ENDDATE']
    df['CREDIT_DURATION_FORESEEN(ENG)'] = df['DAYS_CREDIT_ENDDATE'] - df['DAYS_CREDIT']
    df['CREDIT_DURATION_FACT(ENG)'] = df['DAYS_ENDDATE_FACT'] - df['DAYS_CREDIT']
    df['AMT_CREDIT_PER_DAY_FORESEEN(ENG)'] = df['AMT_CREDIT_SUM']/df['CREDIT_DURATION_FORESEEN(ENG)']
    df['AMT_CREDIT_PER_DAY_FACT(ENG)'] = df['AMT_CREDIT_SUM']/df['CREDIT_DURATION_FACT(ENG)']
    df['AMT_ANNUITY_PER_DAY_FORESEEN(ENG)'] = df['AMT_ANNUITY']/df['CREDIT_DURATION_FORESEEN(ENG)']
    df['AMT_ANNUITY_PER_DAY_FACT(ENG)'] = df['AMT_ANNUITY']/df['CREDIT_DURATION_FACT(ENG)']
    df['PCT_ANNUITY(ENG)'] = df['AMT_ANNUITY']/df['AMT_CREDIT_SUM']
    df['FLAG_OVERDUE(ENG)'] = df['CREDIT_DAY_OVERDUE'].apply(is_positive) | df['AMT_CREDIT_SUM_OVERDUE'].apply(is_positive)
    df['FLAG_ON_DEBT(ENG)'] = df['AMT_CREDIT_SUM_DEBT'].apply(is_positive)
    df_groupby = df.groupby('SK_ID_CURR')
    for feature in ['AMT_CREDIT_SUM_DEBT_PER_REMAINING_DAY(ENG)', 'AMT_CREDIT_SUM_OVERDUE_PER_REMAINING_DAY(ENG)',
                    'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_PAYMENT(ENG)', 'AMT_CREDIT_SUM', 'AMT_ANNUITY']:
        df['ALL_'+feature] = df_groupby[feature].transform('sum')
    df['CNT_OCURRENCES(ENG)'] = df_groupby['SK_ID_CURR'].transform('count')

    get_cnt_diff(df, df_groupby, 'CREDIT_TYPE')

    for feature in ['DAYS_CREDIT_ENDDATE', 'DAYS_CREDIT', 'DAYS_ENDDATE_FACT', 'DAYS_CREDIT_UPDATE']:
        df[feature+'_DAYDIFF(ENG)'] = df_groupby[feature].transform(lambda x: x.sort_values().diff().fillna(0).mean())
        if df[df[feature]>=0].shape[0] > 0:
            df[feature+'_PAST_DAYDIFF(ENG)'] = df_groupby[feature].transform(lambda x: x[x<0].sort_values().diff().fillna(0).mean())
            df[feature+'_FUTURE_DAYDIFF(ENG)'] = df_groupby[feature].transform(lambda x: x[x>=0].sort_values().diff().fillna(0).mean())
            df[feature+'_FUTURE_PAST_DAYDIFF_RATE(ENG)'] = df[feature+'_FUTURE_DAYDIFF(ENG)']/df[feature+'_PAST_DAYDIFF(ENG)']
            df['CNT_'+feature+'_PAST(ENG)'] = df_groupby[feature].transform(lambda x: x[x<0].shape[0])
            df['CNT_'+feature+'_FUTURE(ENG)'] = df_groupby[feature].transform(lambda x: x[x>=0].shape[0])
            df['PCT_'+feature+'_FUTURE(ENG)'] = df['CNT_'+feature+'_FUTURE(ENG)']/df['CNT_OCURRENCES(ENG)']
    df['PCT_PROLONGED(ENG)'] = df_groupby['CNT_CREDIT_PROLONG'].transform(lambda x: x[x>0].shape[0]/x.shape[0]).fillna(0)

    df = pd.get_dummies(df, dummy_na=True, columns=['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE'])

    df_eng = aggr_cols(df, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE', target_cols=df.columns, degree=DEGREE)
    extract_delta(df_eng, df, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE',
                  target_cols=['AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_DEBT_PER_REMAINING_DAY(ENG)',
                               'AMT_CREDIT_SUM_OVERDUE', 'AMT_CREDIT_SUM_OVERDUE_PER_REMAINING_DAY(ENG)',
                               'AMT_CREDIT_PER_DAY_FORESEEN(ENG)', 'AMT_CREDIT_PER_DAY_FACT(ENG)',
                               'AMT_ANNUITY_PER_DAY_FORESEEN(ENG)', 'AMT_ANNUITY_PER_DAY_FACT(ENG)',
                               'FLAG_OVERDUE(ENG)', 'FLAG_ON_DEBT(ENG)', 'AMT_CREDIT_SUM',
                               'AMT_ANNUITY', 'AMT_PAYMENT(ENG)', 'PCT_ANNUITY(ENG)'], degree=DEGREE_DELTA)

    df_eng['PCT_ANNUITY_PER_DAY_FORESEEN(ENG)'] = df_eng['AMT_ANNUITY_PER_DAY_FORESEEN(ENG)']/df_eng['AMT_CREDIT_SUM']
    df_eng['PCT_ANNUITY_PER_DAY_FACT(ENG)'] = df_eng['AMT_ANNUITY_PER_DAY_FACT(ENG)']/df_eng['AMT_CREDIT_SUM']
    df_eng['PCT_CREDIT_DURATION_FACT(ENG)'] = df_eng['CREDIT_DURATION_FACT(ENG)']/df_eng['CREDIT_DURATION_FORESEEN(ENG)']
    df_eng['PCT_CREDIT_SUM_DEBT(ENG)'] = df_eng['AMT_CREDIT_SUM_DEBT']/df_eng['AMT_CREDIT_SUM']
    df_eng['PCT_CREDIT_SUM_DEBT_LIMIT(ENG)'] = df_eng['AMT_CREDIT_SUM_DEBT']/df_eng['AMT_CREDIT_SUM_LIMIT']
    df_eng['PCT_CREDIT_SUM_OVERDUE(ENG)'] = df_eng['AMT_CREDIT_SUM_OVERDUE']/df_eng['AMT_CREDIT_SUM']
    df_eng['PCT_CREDIT_SUM_OVERDUE_LIMIT(ENG)'] = df_eng['AMT_CREDIT_SUM_OVERDUE']/df_eng['AMT_CREDIT_SUM_LIMIT']
    df_eng['PCT_CREDIT_MAX_OVERDUE(ENG)'] = df_eng['AMT_CREDIT_MAX_OVERDUE']/df_eng['AMT_CREDIT_SUM']
    df_eng['PCT_CREDIT_MAX_OVERDUE_LIMIT(ENG)'] = df_eng['AMT_CREDIT_MAX_OVERDUE']/df_eng['AMT_CREDIT_SUM_LIMIT']
    df_eng['PCT_ANNUITY(ENG)'] = df_eng['AMT_ANNUITY']/df_eng['AMT_CREDIT_SUM']
    df_eng['CNT_ACTIVE(ENG)'] = df_eng['CREDIT_ACTIVE_Active']*df_eng['CNT_OCURRENCES(ENG)']
    df_eng['CNT_CLOSED(ENG)'] = df_eng['CREDIT_ACTIVE_Closed']*df_eng['CNT_OCURRENCES(ENG)']
    df_eng['PCT_LIMIT_SUM(ENG)'] = df_eng['AMT_CREDIT_SUM_LIMIT']/df_eng['AMT_CREDIT_SUM']
    df_eng['PCT_OVERDUE_ENDDATE(ENG)'] = df_eng['CREDIT_DAY_OVERDUE']/df_eng['DAYS_CREDIT_ENDDATE']
    df_eng['PCT_OVERDUE_DAYS_CREDIT(ENG)'] = df_eng['CREDIT_DAY_OVERDUE']/df_eng['DAYS_CREDIT']

    del df_eng['MONTHS_BALANCE']

    return df_eng

def eng_bureau_balance(df):

    df['STATUS(ENG)'] = df['STATUS'].map({'0':0, '1':1, '2':2, '3':3, '4':4, '5':6})

    df_groupby = df.groupby('SK_ID_CURR')
    df['CNT_OCURRENCES(ENG)'] = df_groupby['SK_ID_CURR'].transform('count')
    df['ALL_STATUS(ENG)'] = df_groupby['STATUS(ENG)'].transform('sum')
    del df_groupby
    collect()

    df = pd.get_dummies(df, dummy_na=True, columns=['STATUS'])

    df_eng = aggr_cols(df, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE', target_cols=df.columns, degree=DEGREE)
    extract_delta(df_eng, df, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE', target_cols=['STATUS(ENG)'], degree=DEGREE_DELTA)

    df_groupby = df[['SK_ID_CURR', 'SK_ID_BUREAU', 'MONTHS_BALANCE', 'STATUS(ENG)']].groupby(['SK_ID_CURR', 'MONTHS_BALANCE'])
    df_by = df_groupby.sum().reset_index()[['SK_ID_CURR', 'MONTHS_BALANCE']]
    df_by['MONTHLY_STATUS_SUM(ENG)'] = df_groupby.sum().reset_index()['STATUS(ENG)']
    df_by['MONTHLY_STATUS_MEAN(ENG)'] = df_groupby.mean().reset_index()['STATUS(ENG)']
    df_by['CNT_MONTHLY_ACTIVE(ENG)'] = df_groupby['SK_ID_BUREAU'].transform(lambda x: x.nunique())
    del df_groupby
    collect()

    df_by_curr = aggr_cols(df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE', target_cols=df_by.columns, degree=DEGREE)
    extract_delta(df_by_curr, df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE',
                  target_cols=['MONTHLY_STATUS_SUM(ENG)', 'MONTHLY_STATUS_MEAN(ENG)', 'CNT_MONTHLY_ACTIVE(ENG)'], degree=DEGREE_DELTA)
    df_eng = df_eng.merge(df_by_curr.drop(columns='MONTHS_BALANCE'), on='SK_ID_CURR', how='left')
    del df_by_curr, df_by
    collect()

    df_groupby = df[['SK_ID_CURR', 'SK_ID_BUREAU', 'MONTHS_BALANCE', 'STATUS(ENG)']].groupby(['SK_ID_CURR', 'SK_ID_BUREAU'])
    df_by = df_groupby.max().reset_index()[['SK_ID_CURR', 'STATUS(ENG)']].rename(columns={'STATUS(ENG)':'MAXIMAL_STATUS(ENG)'})
    df_by['MONTHS_BALANCE'] = df_groupby.mean().reset_index()['MONTHS_BALANCE']
    df_by_curr = aggr_cols(df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE', target_cols=df_by.columns, degree=DEGREE)
    extract_delta(df_by_curr, df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE', target_cols=['MAXIMAL_STATUS(ENG)'], degree=DEGREE_DELTA)
    df_eng = df_eng.merge(df_by_curr.drop(columns='MONTHS_BALANCE'), on='SK_ID_CURR', how='left')
    del df_groupby, df_by, df_by_curr
    collect()

    return df_eng

def eng_previous_application(df):

    for feature in ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
        df[feature].replace(365243, np.nan, inplace=True)

    df['MONTHS_BALANCE'] = df['DAYS_DECISION']//30
    df['LAST_DUE_CHANGE_DAYDIFF(ENG)'] = df['DAYS_LAST_DUE_1ST_VERSION'] - df['DAYS_LAST_DUE']
    df['AMT_DENIED(ENG)'] = df['AMT_APPLICATION'] - df['AMT_CREDIT']
    df['AMT_GAP_APPLICATION(ENG)'] = df['AMT_APPLICATION'] - df['AMT_GOODS_PRICE']
    df['AMT_GAP_CREDIT(ENG)'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
    df['AMT_CREDIT_PER_MONTH(ENG)'] = df['AMT_CREDIT']/df['CNT_PAYMENT']
    df['AMT_DENIED_PER_MONTH(ENG)'] = df['AMT_DENIED(ENG)']/df['CNT_PAYMENT']
    df['AMT_GAP_APPLICATION_PER_MONTH(ENG)'] = df['AMT_GAP_APPLICATION(ENG)']/df['CNT_PAYMENT']
    df['AMT_GAP_CREDIT_PER_MONTH(ENG)'] = df['AMT_GAP_CREDIT(ENG)']/df['CNT_PAYMENT']
    df['PCT_ANNUITY(ENG)'] = df['AMT_ANNUITY']/df['AMT_CREDIT']

    df['FLAG_LAST_APPL_PER_CONTRACT'].replace(['N', 'Y'], [0, 1], inplace=True)

    df_groupby = df.groupby('SK_ID_CURR')

    df['CNT_OCURRENCES(ENG)'] = df_groupby['SK_ID_CURR'].transform('count')
    df['ALL_AMT_CREDIT(ENG)'] = df_groupby['AMT_CREDIT'].transform('sum')
    df['ALL_AMT_ANNUITY(ENG)'] = df_groupby['AMT_ANNUITY'].transform('sum')
    df['ALL_AMT_DOWN_PAYMENT(ENG)'] = df_groupby['AMT_DOWN_PAYMENT'].transform('sum')

    df['SELLERPLACE_AREA_STD(ENG)'] = df_groupby['SELLERPLACE_AREA'].transform('std')
    df['SELLERPLACE_AREA_NUNIQUE(ENG)'] = df_groupby['SELLERPLACE_AREA'].transform(lambda x: x.nunique())

    categorical_features = ['NAME_PRODUCT_TYPE', 'NAME_YIELD_GROUP', 'NAME_SELLER_INDUSTRY', 'NAME_CONTRACT_STATUS',
            'NAME_PORTFOLIO', 'NAME_PAYMENT_TYPE', 'NAME_GOODS_CATEGORY', 'NAME_CONTRACT_TYPE', 'NAME_CLIENT_TYPE',
            'NAME_CASH_LOAN_PURPOSE', 'CODE_REJECT_REASON', 'CHANNEL_TYPE', 'WEEKDAY_APPR_PROCESS_START',
            'NAME_TYPE_SUITE', 'PRODUCT_COMBINATION']

    for feature in ['DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION', 'DAYS_DECISION']:
        df[feature+'_DAYDIFF(ENG)'] = df_groupby[feature].transform(lambda x: x.sort_values().diff().fillna(0).mean())
        if df[df[feature]>=0].shape[0] > 0:
            df[feature+'_PAST_DAYDIFF(ENG)'] = df_groupby[feature].transform(lambda x: x[x<0].sort_values().diff().fillna(0).mean())
            df[feature+'_FUTURE_DAYDIFF(ENG)'] = df_groupby[feature].transform(lambda x: x[x>=0].sort_values().diff().fillna(0).mean())
            df[feature+'_FUTURE_PAST_DAYDIFF_RATE(ENG)'] = df[feature+'_FUTURE_DAYDIFF(ENG)']/df[feature+'_PAST_DAYDIFF(ENG)']
            df['CNT_'+feature+'_PAST(ENG)'] = df_groupby[feature].transform(lambda x: x[x<0].shape[0])
            df['CNT_'+feature+'_FUTURE(ENG)'] = df_groupby[feature].transform(lambda x: x[x>=0].shape[0])
            df['PCT_'+feature+'_FUTURE(ENG)'] = df['CNT_'+feature+'_FUTURE(ENG)']/df['CNT_OCURRENCES(ENG)']

    for feature in categorical_features:
        get_cnt_diff(df, df_groupby, feature)

    del df_groupby
    collect()

    df = pd.get_dummies(df, dummy_na=True, columns=categorical_features)

    df_eng = aggr_cols(df, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE', target_cols=df.columns, degree=DEGREE)
    extract_delta(df_eng, df, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE',
                  target_cols=['AMT_DENIED(ENG)', 'AMT_CREDIT_PER_MONTH(ENG)', 'AMT_DENIED_PER_MONTH(ENG)',
                               'PCT_ANNUITY(ENG)', 'AMT_CREDIT', 'AMT_APPLICATION', 'AMT_ANNUITY',
                               'AMT_DOWN_PAYMENT'], degree=DEGREE_DELTA)

    df_eng['PCT_ANNUITY(ENG)'] = df_eng['AMT_ANNUITY']/df_eng['AMT_CREDIT']
    df_eng['PCT_DENIED(ENG)'] = df_eng['AMT_DENIED(ENG)']/df_eng['AMT_APPLICATION']
    df_eng['PCT_DOWN_PAYMENT(ENG)'] = df_eng['AMT_DOWN_PAYMENT']/df_eng['AMT_CREDIT']
    df_eng['PCT_GOODS_PRICE_APPLICATION(ENG)'] = df_eng['AMT_GOODS_PRICE']/df_eng['AMT_APPLICATION']
    df_eng['PCT_GOODS_PRICE_CREDIT(ENG)'] = df_eng['AMT_GOODS_PRICE']/df_eng['AMT_CREDIT']

    del df_eng['MONTHS_BALANCE']

    return df_eng

def eng_pos_cash_balance(df):

    df_groupby = df.groupby('SK_ID_CURR')
    get_cnt_diff(df, df_groupby, 'NAME_CONTRACT_STATUS')
    df['CNT_OCURRENCES(ENG)'] = df_groupby['SK_ID_CURR'].transform('count')
    del df_groupby
    collect()

    df = pd.get_dummies(df, columns=['NAME_CONTRACT_STATUS'])

    df_groupby = df.groupby(['SK_ID_CURR', 'SK_ID_PREV'])
    cnt_instalment_means = df_groupby['CNT_INSTALMENT'].transform('mean')
    df['PCT_INSTALMENT_FUTURE(ENG)'] = df_groupby['CNT_INSTALMENT_FUTURE'].transform('mean')/cnt_instalment_means
    df['PCT_LENGTH_INSTALMENT(ENG)'] = (df_groupby['CNT_INSTALMENT'].transform('count')-1)/cnt_instalment_means
    del cnt_instalment_means, df_groupby
    collect()

    df_eng = aggr_cols(df, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE', target_cols=df.columns, degree=DEGREE)
    extract_delta(df_eng, df, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE',
                  target_cols=['PCT_INSTALMENT_FUTURE(ENG)', 'PCT_LENGTH_INSTALMENT(ENG)', 'SK_DPD', 'SK_DPD_DEF'], degree=DEGREE_DELTA)

    df_groupby = df[['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF',
                     'NAME_CONTRACT_STATUS_Completed']].groupby(['SK_ID_CURR', 'SK_ID_PREV'])

    df_by = df_groupby.max().reset_index().rename(columns={'SK_DPD':'MAXIMAL_SK_DPD(ENG)',
                                                           'SK_DPD_DEF':'MAXIMAL_SK_DPD_DEF(ENG)',
                                                           'NAME_CONTRACT_STATUS_Completed':'FLAG_COMPLETED(ENG)'})

    df_active = df[df['MONTHS_BALANCE']==-1][['SK_ID_CURR', 'SK_ID_PREV', 'NAME_CONTRACT_STATUS_Active']]
    df_active = df_active.rename(columns={'NAME_CONTRACT_STATUS_Active': 'FLAG_ACTIVE(ENG)'})
    df_by = df_by.merge(df_active, on=['SK_ID_CURR', 'SK_ID_PREV'], how='left')
    df_by['FLAG_ACTIVE(ENG)'].fillna(0, inplace=True)
    df_by_curr_groupby = df_by.groupby('SK_ID_CURR')
    df_by['PCT_ACTIVE(ENG)'] = df_by_curr_groupby['FLAG_ACTIVE(ENG)'].transform('mean')
    df_by['CNT_ACTIVE(ENG)'] = df_by_curr_groupby['FLAG_ACTIVE(ENG)'].transform('sum')
    df_by['PCT_COMPLETED(ENG)'] = df_by_curr_groupby['FLAG_COMPLETED(ENG)'].transform('mean')
    df_by['CNT_COMPLETED(ENG)'] = df_by_curr_groupby['FLAG_COMPLETED(ENG)'].transform('sum')
    del df_active, df_by_curr_groupby, df_by['FLAG_COMPLETED(ENG)']
    collect()

    df_by['MONTHS_BALANCE'] = df_groupby.mean().reset_index()['MONTHS_BALANCE']
    df_by_curr = aggr_cols(df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE', target_cols=df_by.columns, degree=DEGREE)
    extract_delta(df_by_curr, df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE',
                  target_cols=['MAXIMAL_SK_DPD(ENG)', 'MAXIMAL_SK_DPD_DEF(ENG)', 'FLAG_ACTIVE(ENG)'], degree=DEGREE_DELTA)
    df_eng = df_eng.merge(df_by_curr.drop(columns='MONTHS_BALANCE'), on='SK_ID_CURR', how='left')
    del df_by, df_by_curr, df_groupby
    collect()

    df_groupby = df[['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF']].groupby(['SK_ID_CURR', 'MONTHS_BALANCE'])
    df_by = df_groupby.sum().reset_index()[['SK_ID_CURR', 'MONTHS_BALANCE']]
    sums = df_groupby.sum().reset_index()
    means = df_groupby.mean().reset_index()
    df_by['MONTHLY_SK_DPD_SUM(ENG)'] = sums['SK_DPD']
    df_by['MONTHLY_SK_DPD_MEAN(ENG)'] = means['SK_DPD']
    df_by['MONTHLY_SK_DPD_DEF_SUM(ENG)'] = sums['SK_DPD_DEF']
    df_by['MONTHLY_SK_DPD_DEF_MEAN(ENG)'] = means['SK_DPD_DEF']
    df_by['CNT_MONTHLY_ACTIVE(ENG)'] = df_groupby['SK_ID_PREV'].transform(lambda x: x.nunique())
    del df_groupby, sums, means
    collect()

    df_by_curr = aggr_cols(df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE', target_cols=df_by.columns, degree=DEGREE)
    extract_delta(df_by_curr, df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE',
                  target_cols=['MONTHLY_SK_DPD_SUM(ENG)', 'MONTHLY_SK_DPD_MEAN(ENG)',
                  'MONTHLY_SK_DPD_DEF_SUM(ENG)', 'MONTHLY_SK_DPD_DEF_MEAN(ENG)',
                  'CNT_MONTHLY_ACTIVE(ENG)'], degree=DEGREE_DELTA)
    df_eng = df_eng.merge(df_by_curr.drop(columns='MONTHS_BALANCE'), on='SK_ID_CURR', how='left')
    del df_by_curr, df_by
    collect()

    return df_eng

def eng_installments_payments(df):

    df['MONTHS_BALANCE'] = df['DAYS_INSTALMENT']//30
    df['PAYMENT_DELAY(ENG)'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
    df['AMT_INSTALMENT_DEBT(ENG)'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
    df['FLAG_OVERDUE(ENG)'] = df['PAYMENT_DELAY(ENG)'].apply(is_positive)
    df['FLAG_ON_DEBT(ENG)'] = df['AMT_INSTALMENT_DEBT(ENG)'].apply(is_positive)
    df['PCT_PAYMENT(ENG)'] = df['AMT_PAYMENT']/df['AMT_INSTALMENT']

    df_groupby = df.groupby('SK_ID_CURR')
    df['CNT_OCURRENCES(ENG)'] = df_groupby['SK_ID_CURR'].transform('count')
    df['ALL_AMT_PAYMENT(ENG)'] = df_groupby['AMT_PAYMENT'].transform('sum')
    df['ALL_AMT_INSTALMENT_DEBT(ENG)'] = df_groupby['AMT_INSTALMENT_DEBT(ENG)'].transform('sum')
    df['ALL_PAYMENT_DELAY(ENG)'] = df_groupby['PAYMENT_DELAY(ENG)'].transform('sum')

    for feature in ['DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT']:
        df[feature+'_DAYDIFF(ENG)'] = df_groupby[feature].transform(lambda x: x.sort_values().diff().fillna(0).mean())
    del df_groupby
    collect()

    df_groupby = df[['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_ENTRY_PAYMENT']].groupby(['SK_ID_CURR', 'SK_ID_PREV'])

    df['DAYS_ENTRY_PAYMENT_DAYDIFF_BYPREV_MEAN(ENG)'] = df_groupby['DAYS_ENTRY_PAYMENT'].transform(lambda x: x.sort_values().diff().fillna(30).mean())
    df['DAYS_ENTRY_PAYMENT_DAYDIFF_BYPREV_STD(ENG)'] = df_groupby['DAYS_ENTRY_PAYMENT'].transform(lambda x: x.sort_values().diff().fillna(30).std())

    del df_groupby
    collect()

    df_eng = aggr_cols(df, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE', target_cols=df.columns, degree=DEGREE)
    extract_delta(df_eng, df, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE',
                  target_cols=['PAYMENT_DELAY(ENG)', 'AMT_INSTALMENT_DEBT(ENG)',
                               'PCT_PAYMENT(ENG)', 'FLAG_OVERDUE(ENG)', 'FLAG_ON_DEBT(ENG)'], degree=DEGREE_DELTA)

    df_eng['PCT_PAYMENT(ENG)'] = df_eng['AMT_PAYMENT']/df_eng['AMT_INSTALMENT']

    df_groupby = df[['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'PAYMENT_DELAY(ENG)',
                     'AMT_INSTALMENT_DEBT(ENG)', 'FLAG_OVERDUE(ENG)', 'FLAG_ON_DEBT(ENG)',
                     'PCT_PAYMENT(ENG)', 'AMT_PAYMENT']].groupby(['SK_ID_CURR', 'SK_ID_PREV'])

    df_by = df_groupby.max().reset_index().rename(columns={'PAYMENT_DELAY(ENG)':'MAXIMAL_PAYMENT_DELAY(ENG)',
                                                           'AMT_INSTALMENT_DEBT(ENG)':'MAXIMAL_AMT_INSTALMENT_DEBT(ENG)',
                                                           'PCT_PAYMENT(ENG)':'MAXIMAL_PCT_PAYMENT(ENG)',
                                                           'FLAG_OVERDUE(ENG)':'PCT_OVERDUE(ENG)',
                                                           'FLAG_ON_DEBT(ENG)':'PCT_ON_DEBT(ENG)'
                                                           }).drop(columns=['AMT_PAYMENT'])

    df_by['MONTHS_BALANCE'] = df_groupby.mean().reset_index()['MONTHS_BALANCE']

    df_by_curr = aggr_cols(df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE', target_cols=df_by.columns, degree=DEGREE)

    extract_delta(df_by_curr, df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE',
                  target_cols=['MAXIMAL_PAYMENT_DELAY(ENG)', 'MAXIMAL_AMT_INSTALMENT_DEBT(ENG)', 'MAXIMAL_PCT_PAYMENT(ENG)',
                               'PCT_OVERDUE(ENG)', 'PCT_ON_DEBT(ENG)'], degree=DEGREE_DELTA)

    df_eng = df_eng.merge(df_by_curr.drop(columns='MONTHS_BALANCE'), on='SK_ID_CURR', how='left')

    del df_by, df_by_curr
    collect()

    df_by = df_groupby.min().reset_index().rename(columns={'PAYMENT_DELAY(ENG)':'MINIMAL_PAYMENT_DELAY(ENG)',
                                                           'AMT_INSTALMENT_DEBT(ENG)':'MINIMAL_AMT_INSTALMENT_DEBT(ENG)',
                                                           'PCT_PAYMENT(ENG)':'MINIMAL_PCT_PAYMENT(ENG)'
                                                           }).drop(columns=['AMT_PAYMENT', 'FLAG_OVERDUE(ENG)', 'FLAG_ON_DEBT(ENG)'])

    df_by['MONTHS_BALANCE'] = df_groupby.mean().reset_index()['MONTHS_BALANCE']

    df_by_curr = aggr_cols(df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE', target_cols=df_by.columns, degree=DEGREE)

    extract_delta(df_by_curr, df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE',
                  target_cols=['MINIMAL_PAYMENT_DELAY(ENG)', 'MINIMAL_AMT_INSTALMENT_DEBT(ENG)',
                               'MINIMAL_PCT_PAYMENT(ENG)'], degree=DEGREE_DELTA)

    df_eng = df_eng.merge(df_by_curr.drop(columns='MONTHS_BALANCE'), on='SK_ID_CURR', how='left')

    del df_by, df_by_curr, df_groupby
    collect()

    df_groupby = df[['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'PAYMENT_DELAY(ENG)',
                     'AMT_INSTALMENT', 'AMT_PAYMENT']].groupby(['SK_ID_CURR', 'MONTHS_BALANCE'])
    df_by = df_groupby.sum().reset_index()[['SK_ID_CURR', 'MONTHS_BALANCE']]
    sums = df_groupby.sum().reset_index()
    means = df_groupby.mean().reset_index()
    df_by['MONTHLY_PAYMENT_DELAY_SUM(ENG)'] = sums['PAYMENT_DELAY(ENG)']
    df_by['MONTHLY_PAYMENT_DELAY_MEAN(ENG)'] = means['PAYMENT_DELAY(ENG)']
    df_by['MONTHLY_AMT_INSTALMENT_SUM(ENG)'] = sums['AMT_INSTALMENT']
    df_by['MONTHLY_AMT_PAYMENT_SUM(ENG)'] = means['AMT_PAYMENT']
    df_by['CNT_MONTHLY_ACTIVE(ENG)'] = df_groupby['SK_ID_PREV'].transform(lambda x: x.nunique())
    del df_groupby, sums, means
    collect()

    df_by_curr = aggr_cols(df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE', target_cols=df_by.columns, degree=DEGREE)
    extract_delta(df_by_curr, df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE',
                  target_cols=['MONTHLY_PAYMENT_DELAY_SUM(ENG)', 'MONTHLY_PAYMENT_DELAY_MEAN(ENG)',
                  'MONTHLY_AMT_INSTALMENT_SUM(ENG)', 'MONTHLY_AMT_PAYMENT_SUM(ENG)',
                  'CNT_MONTHLY_ACTIVE(ENG)'], degree=DEGREE_DELTA)
    df_eng = df_eng.merge(df_by_curr.drop(columns='MONTHS_BALANCE'), on='SK_ID_CURR', how='left')
    del df_by_curr, df_by
    collect()

    del df_eng['MONTHS_BALANCE']

    return df_eng

def eng_credit_card_balance(df):

    df.fillna(0, inplace=True)

    df = pd.get_dummies(df, columns=['NAME_CONTRACT_STATUS'])

    df_groupby = df.groupby('SK_ID_CURR')
    df['CNT_OCURRENCES(ENG)'] = df_groupby['SK_ID_CURR'].transform('count')
    del df_groupby
    collect()

    df['PCT_BLANCE(ENG)'] = df['AMT_BALANCE']/df['AMT_CREDIT_LIMIT_ACTUAL']

    df_groupby = df[['SK_ID_CURR', 'SK_ID_PREV', 'CNT_INSTALMENT_MATURE_CUM']].groupby(['SK_ID_CURR', 'SK_ID_PREV'])

    df['CNT_INSTALMENT_MATURE_CUM_DIFF_BYPREV_MEAN(ENG)'] = df_groupby['CNT_INSTALMENT_MATURE_CUM'].transform(lambda x: x.sort_values().diff().fillna(1).mean())
    df['CNT_INSTALMENT_MATURE_CUM_DIFF_BYPREV_STD(ENG)'] = df_groupby['CNT_INSTALMENT_MATURE_CUM'].transform(lambda x: x.sort_values().diff().fillna(1).std())

    del df_groupby
    collect()

    df_eng = aggr_cols(df, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE', target_cols=df.columns, degree=DEGREE)
    extract_delta(df_eng, df, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE',
                  target_cols=['PCT_BLANCE(ENG)', 'SK_DPD', 'SK_DPD_DEF'], degree=DEGREE_DELTA)

    df_groupby = df[['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF',
                     'AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'PCT_BLANCE(ENG)',
                     'NAME_CONTRACT_STATUS_Completed']].groupby(['SK_ID_CURR', 'SK_ID_PREV'])

    df_by = df_groupby.max().reset_index().rename(columns={'SK_DPD':'MAXIMAL_SK_DPD(ENG)',
                                                           'SK_DPD_DEF':'MAXIMAL_SK_DPD_DEF(ENG)',
                                                           'AMT_BALANCE':'MAXIMAL_AMT_BALANCE(ENG)',
                                                           'AMT_CREDIT_LIMIT_ACTUAL':'MAXIMAL_AMT_CREDIT_LIMIT_ACTUAL(ENG)',
                                                           'PCT_BLANCE(ENG)':'MAXIMAL_PCT_BLANCE(ENG)',
                                                           'NAME_CONTRACT_STATUS_Completed':'FLAG_COMPLETED(ENG)'})

    df_active = df[df['MONTHS_BALANCE']==-1][['SK_ID_CURR', 'SK_ID_PREV', 'NAME_CONTRACT_STATUS_Active']]
    df_active = df_active.rename(columns={'NAME_CONTRACT_STATUS_Active': 'FLAG_ACTIVE(ENG)'})
    df_by = df_by.merge(df_active, on=['SK_ID_CURR', 'SK_ID_PREV'], how='left')
    df_by['FLAG_ACTIVE(ENG)'].fillna(0, inplace=True)
    df_by_curr_groupby = df_by.groupby('SK_ID_CURR')
    df_by['CNT_ACTIVE(ENG)'] = df_by_curr_groupby['FLAG_ACTIVE(ENG)'].transform('sum')
    df_by['CNT_COMPLETED(ENG)'] = df_by_curr_groupby['FLAG_COMPLETED(ENG)'].transform('sum')
    del df_active, df_by_curr_groupby, df_by['FLAG_ACTIVE(ENG)'], df_by['FLAG_COMPLETED(ENG)']
    collect()

    df_by['MONTHS_BALANCE'] = df_groupby.mean().reset_index()['MONTHS_BALANCE']
    df_by_curr = aggr_cols(df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE', target_cols=df_by.columns, degree=DEGREE)
    extract_delta(df_by_curr, df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE',
                  target_cols=['MAXIMAL_SK_DPD(ENG)', 'MAXIMAL_SK_DPD_DEF(ENG)',
                               'MAXIMAL_AMT_CREDIT_LIMIT_ACTUAL(ENG)', 'MAXIMAL_PCT_BLANCE(ENG)'], degree=DEGREE_DELTA)
    df_eng = df_eng.merge(df_by_curr.drop(columns='MONTHS_BALANCE'), on='SK_ID_CURR', how='left')
    del df_by, df_by_curr
    collect()

    df_by = df_groupby.min().reset_index().rename(columns={'SK_DPD':'MINIMAL_SK_DPD(ENG)',
                                                           'SK_DPD_DEF':'MINIMAL_SK_DPD_DEF(ENG)',
                                                           'AMT_BALANCE':'MINIMAL_AMT_BALANCE(ENG)',
                                                           'AMT_CREDIT_LIMIT_ACTUAL':'MINIMAL_AMT_CREDIT_LIMIT_ACTUAL(ENG)',
                                                           'PCT_BLANCE(ENG)':'MINIMAL_PCT_BLANCE(ENG)'}).drop(columns=[
                                                               'NAME_CONTRACT_STATUS_Completed'
                                                           ])

    df_by['MONTHS_BALANCE'] = df_groupby.mean().reset_index()['MONTHS_BALANCE']
    df_by_curr = aggr_cols(df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE', target_cols=df_by.columns, degree=DEGREE)
    extract_delta(df_by_curr, df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE',
                  target_cols=['MINIMAL_SK_DPD(ENG)', 'MINIMAL_SK_DPD_DEF(ENG)',
                               'MINIMAL_AMT_CREDIT_LIMIT_ACTUAL(ENG)', 'MINIMAL_PCT_BLANCE(ENG)'], degree=DEGREE_DELTA)
    df_eng = df_eng.merge(df_by_curr.drop(columns='MONTHS_BALANCE'), on='SK_ID_CURR', how='left')

    del df_by, df_by_curr, df_groupby
    collect()

    df_eng['PCT_BLANCE(ENG)'] = df_eng['AMT_BALANCE']/df_eng['AMT_CREDIT_LIMIT_ACTUAL']

    df_groupby = df[['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'AMT_BALANCE',
                     'AMT_CREDIT_LIMIT_ACTUAL', 'SK_DPD', 'SK_DPD_DEF']].groupby(['SK_ID_CURR', 'MONTHS_BALANCE'])

    df_by = df_groupby.sum().reset_index()[['SK_ID_CURR', 'MONTHS_BALANCE']]
    sums = df_groupby.sum().reset_index()
    df_by['MONTHLY_AMT_BALANCE_SUM(ENG)'] = sums['AMT_BALANCE']
    df_by['MONTHLY_AMT_CREDIT_LIMIT_ACTUAL_SUM(ENG)'] = sums['AMT_CREDIT_LIMIT_ACTUAL']
    df_by['PCT_MONTHLY_AMT_BALANCE_SUM(ENG)'] = df_by['MONTHLY_AMT_BALANCE_SUM(ENG)']/df_by['MONTHLY_AMT_CREDIT_LIMIT_ACTUAL_SUM(ENG)']
    df_by['MONTHLY_SK_DPD_SUM(ENG)'] = sums['SK_DPD']
    df_by['MONTHLY_SK_DPD_DEF_SUM(ENG)'] = sums['SK_DPD_DEF']
    del df_groupby, sums
    collect()

    df_by_curr = aggr_cols(df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE', target_cols=df_by.columns, degree=DEGREE)

    extract_delta(df_by_curr, df_by, by=['SK_ID_CURR'], past_col='MONTHS_BALANCE',
                  target_cols=['MONTHLY_AMT_BALANCE_SUM(ENG)', 'MONTHLY_AMT_CREDIT_LIMIT_ACTUAL_SUM(ENG)',
                  'PCT_MONTHLY_AMT_BALANCE_SUM(ENG)', 'MONTHLY_SK_DPD_DEF_SUM(ENG)'], degree=DEGREE_DELTA)

    df_eng = df_eng.merge(df_by_curr.drop(columns='MONTHS_BALANCE'), on='SK_ID_CURR', how='left')
    del df_by_curr, df_by
    collect()

    return df_eng

def eng_application(df):

    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['PCT_EMPLOYED(ENG)'] = df['DAYS_EMPLOYED']/df['DAYS_BIRTH']
    df['PCT_CHILDREN'] = df['CNT_CHILDREN']/df['CNT_FAM_MEMBERS']
    df['PCT_ANNUITY_CREDIT(ENG)'] = df['AMT_ANNUITY']/df['AMT_CREDIT']
    df['PCT_ANNUITY_INCOME(ENG)'] = df['AMT_ANNUITY']/df['AMT_INCOME_TOTAL']
    df['PCT_CREDIT_INCOME(ENG)'] = df['AMT_CREDIT']/df['AMT_INCOME_TOTAL']
    df['PCT_GOODS_PRICE_INCOME(ENG)'] = df['AMT_GOODS_PRICE']/df['AMT_INCOME_TOTAL']
    df['PCT_GOODS_PRICE_CREDIT(ENG)'] = df['AMT_GOODS_PRICE']/df['AMT_CREDIT']
    df['PCT_GOODS_PRICE_ANNUITY(ENG)'] = df['AMT_GOODS_PRICE']/df['AMT_ANNUITY']
    df['AMT_INCOME_PER_CHILD(ENG)'] = df['AMT_INCOME_TOTAL']/df['CNT_CHILDREN']
    df['PCT_INCOME_PER_CHILD_CREDIT(ENG)'] = df['AMT_INCOME_PER_CHILD(ENG)']/df['AMT_CREDIT']
    df['PCT_INCOME_PER_CHILD_ANNUITY(ENG)'] = df['AMT_INCOME_PER_CHILD(ENG)']/df['AMT_ANNUITY']
    df['AMT_INCOME_PER_FAM_MEMBER(ENG)'] = df['AMT_INCOME_TOTAL']/df['CNT_FAM_MEMBERS']
    df['PCT_INCOME_PER_FAM_MEMBER_CREDIT(ENG)'] = df['AMT_INCOME_PER_FAM_MEMBER(ENG)']/df['AMT_CREDIT']
    df['PCT_INCOME_PER_FAM_MEMBER_ANNUITY(ENG)'] = df['AMT_INCOME_PER_FAM_MEMBER(ENG)']/df['AMT_ANNUITY']
    df['CNT_ADULTS(ENG)'] = df['CNT_FAM_MEMBERS']-df['CNT_CHILDREN']
    df['AMT_INCOME_PER_ADULT(ENG)'] = df['AMT_INCOME_TOTAL']/df['CNT_ADULTS(ENG)']
    df['PCT_INCOME_PER_ADULT_CREDIT(ENG)'] = df['AMT_INCOME_PER_ADULT(ENG)']/df['AMT_CREDIT']
    df['PCT_INCOME_PER_ADULT_ANNUITY(ENG)'] = df['AMT_INCOME_PER_ADULT(ENG)']/df['AMT_ANNUITY']
    df['RELATIVE_DEF_60_CNT_SOCIAL_CIRCLE(ENG)'] = df['DEF_60_CNT_SOCIAL_CIRCLE']/df['REGION_POPULATION_RELATIVE']
    df['RELATIVE_DEF_30_CNT_SOCIAL_CIRCLE(ENG)'] = df['DEF_30_CNT_SOCIAL_CIRCLE']/df['REGION_POPULATION_RELATIVE']
    df['RELATIVE_OBS_60_CNT_SOCIAL_CIRCLE(ENG)'] = df['OBS_60_CNT_SOCIAL_CIRCLE']/df['REGION_POPULATION_RELATIVE']
    df['RELATIVE_OBS_30_CNT_SOCIAL_CIRCLE(ENG)'] = df['OBS_30_CNT_SOCIAL_CIRCLE']/df['REGION_POPULATION_RELATIVE']

    ext1, ext2, ext3 = 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'

    for ext_group in [[ext1, ext2], [ext1, ext3], [ext2, ext3], [ext1, ext2, ext3]]:
        prod_feature = '_TIMES_'.join(ext_group)+'(ENG)'
        sum_feature = '_PLUS_'.join(ext_group)+'(ENG)'
        df[prod_feature] = 1
        df[sum_feature] = 0
        for ext in ext_group:
            df[prod_feature] *= df[ext]
            df[sum_feature] += df[ext]

    periods = ['HOUR', 'DAY', 'WEEK', 'MON', 'QRT', 'YEAR']
    for i in range(len(periods)-1):
        for j in range(i+1, len(periods)):
            minor, major = periods[i], periods[j]
            df['PCT_'+minor+'_'+major+'(ENG)'] = df['AMT_REQ_CREDIT_BUREAU_'+minor]/df['AMT_REQ_CREDIT_BUREAU_'+major]

    for feature in ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[feature].replace(['N', 'Y'], [0, 1], inplace=True)

    df.rename(columns={'TOTALAREA_MODE':'TOTALAREA', 'EMERGENCYSTATE_MODE': 'EMERGENCYSTATE'}, inplace=True)

    df['EMERGENCYSTATE'].replace(['No', 'Yes'], [0, 1], inplace=True)

    living_features = ['COMMONAREA', 'NONLIVINGAPARTMENTS', 'LIVINGAPARTMENTS', 'FLOORSMIN', 'YEARS_BUILD',
                       'LANDAREA', 'BASEMENTAREA', 'NONLIVINGAREA', 'ELEVATORS', 'APARTMENTS', 'ENTRANCES',
                       'LIVINGAREA', 'FLOORSMAX', 'YEARS_BEGINEXPLUATATION', 'TOTALAREA', 'EMERGENCYSTATE']

    for feature in living_features:
        if feature not in ['TOTALAREA', 'EMERGENCYSTATE']:
            df[feature] = (df[feature+'_MODE'] + df[feature+'_MEDI'] + df[feature+'_AVG'])/3
            del df[feature+'_MODE'], df[feature+'_MEDI'], df[feature+'_AVG']

    df = pd.get_dummies(df, dummy_na=True, columns=['FONDKAPREMONT_MODE', 'WALLSMATERIAL_MODE', 'HOUSETYPE_MODE', 'OCCUPATION_TYPE',
                            'NAME_TYPE_SUITE', 'ORGANIZATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'CODE_GENDER', 'NAME_CONTRACT_TYPE',
                            'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE'])

    del df['CODE_GENDER_F']

    return df

def process_bureau():

    print('processing bureau')

    start = time()

    bureau = pd.read_csv('input/bureau.csv')
    bureau = bureau[bureau['AMT_CREDIT_SUM']>0].reset_index(drop=True)
    place_nulls(bureau)

    bureau_balance = pd.read_csv('input/bureau_balance.csv')
    place_nulls(bureau_balance)
    bureau_balance = bureau_balance.merge(bureau[['SK_ID_CURR', 'SK_ID_BUREAU']].drop_duplicates(subset=['SK_ID_BUREAU']),
                                            on='SK_ID_BUREAU',
                                            how='inner')
    del bureau['SK_ID_BUREAU']

    bureau_balance = eng_bureau_balance(bureau_balance)
    rename_columns(bureau_balance, suffix='(BUR_BAL)')

    bureau = eng_bureau(bureau)
    rename_columns(bureau, suffix='(BUREAU)')

    bureau = bureau.merge(bureau_balance, on='SK_ID_CURR', how='left')
    dump(bureau, open('intermediary/bureau.pkl', 'wb'))

    print('finished bureau ({:.2f}m)'.format((time()-start)/60))

def process_previous_application():

    print('processing previous_application')

    start = time()

    previous_application = pd.read_csv('input/previous_application.csv').drop(columns=['SK_ID_PREV'])
    previous_application = previous_application[previous_application['AMT_CREDIT']>0].reset_index(drop=True)
    place_nulls(previous_application)
    previous_application = eng_previous_application(previous_application)
    rename_columns(previous_application, suffix='(PREV_APP)')
    dump(previous_application, open('intermediary/previous_application.pkl', 'wb'))

    print('finished previous_application ({:.2f}m)'.format((time()-start)/60))

def process_pos_cash_balance():

    print('processing pos_cash_balance')

    start = time()

    pos_cash_balance = pd.read_csv('input/POS_CASH_balance.csv')
    place_nulls(pos_cash_balance)
    pos_cash_balance = eng_pos_cash_balance(pos_cash_balance)
    rename_columns(pos_cash_balance, suffix='(POS_CASH)')
    dump(pos_cash_balance, open('intermediary/pos_cash_balance.pkl', 'wb'))

    print('finished pos_cash_balance ({:.2f}m)'.format((time()-start)/60))

def process_installments_payments():

    print('processing installments_payments')

    start = time()

    installments_payments = pd.read_csv('input/installments_payments.csv')
    place_nulls(installments_payments)
    installments_payments = eng_installments_payments(installments_payments)
    rename_columns(installments_payments, suffix='(INST_PAYM)')
    dump(installments_payments, open('intermediary/installments_payments.pkl', 'wb'))

    print('finished installments_payments ({:.2f}m)'.format((time()-start)/60))

def process_credit_card_balance():

    print('processing credit_card_balance')

    start = time()

    credit_card_balance = pd.read_csv('input/credit_card_balance.csv',
                                      usecols=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'AMT_BALANCE',
                                               'AMT_CREDIT_LIMIT_ACTUAL', 'CNT_INSTALMENT_MATURE_CUM',
                                               'NAME_CONTRACT_STATUS', 'SK_DPD', 'SK_DPD_DEF'])
    place_nulls(credit_card_balance)
    credit_card_balance = eng_credit_card_balance(credit_card_balance)
    rename_columns(credit_card_balance, suffix='(CRED_CARD)')
    dump(credit_card_balance, open('intermediary/credit_card_balance.pkl', 'wb'))

    print('finished credit_card_balance ({:.2f}m)'.format((time()-start)/60))

def process_application():

    print('processing application')

    start = time()

    application_train = pd.read_csv('input/application_train.csv')
    application_test = pd.read_csv('input/application_test.csv')
    application_test['TARGET'] = 2

    columns = list(application_train.columns)
    columns.remove('SK_ID_CURR')
    columns.remove('TARGET')
    columns = ['SK_ID_CURR'] + columns + ['TARGET']

    application_train = application_train[columns]
    application_test = application_test[columns]

    application = pd.concat([application_train, application_test], ignore_index=True)
    del application_train, application_test
    collect()

    place_nulls(application)
    application = eng_application(application)

    rename_columns(application, suffix='(CURR_APP)')
    dump(application, open('intermediary/application.pkl', 'wb'))

    print('finished application ({:.2f}m)'.format((time()-start)/60))

###################

available_dfs = ['application', 'bureau', 'previous_application',
                 'pos_cash_balance', 'installments_payments', 'credit_card_balance']

process_functions = dict(
    application = process_application,
    bureau = process_bureau,
    previous_application = process_previous_application,
    pos_cash_balance = process_pos_cash_balance,
    installments_payments = process_installments_payments,
    credit_card_balance = process_credit_card_balance
)

if len(sys.argv) == 2:
    if sys.argv[1] not in available_dfs:
        print('invalid dataframe. available dataframes are:')
        print('\t*'+'\n\t*'.join(available_dfs))
    else:
        process_functions[sys.argv[1]]()
elif len(sys.argv) == 1:
    for key in process_functions:
        process_functions[key]()
else:
    print('invalid input. enter either none or one of the available dataframes:')
    print('\t*'+'\n\t*'.join(available_dfs))
