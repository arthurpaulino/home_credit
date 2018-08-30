from data_aux_functions import *
from pickle import load, dump
from parameters import *
from gc import collect
from time import time
import pandas as pd

def pos_merge_eng(df):
    for feature in ['AMT_CREDIT_SUM_DEBT_PER_REMAINING_DAY(ENG)', 'AMT_CREDIT_SUM_OVERDUE_PER_REMAINING_DAY(ENG)',
                    'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_PAYMENT(ENG)', 'AMT_CREDIT_SUM', 'AMT_ANNUITY']:
        for prefix in ['', 'ALL_']:
            prefix_feature = prefix+feature
            new_name = prefix_feature.split('(ENG)')[0]
            df['PCT_'+new_name+'_INCOME(ENG)(CURR_APP)(BUREAU)'] = df[prefix_feature+'(BUREAU)']/df['AMT_INCOME_TOTAL(CURR_APP)']
            df['PCT_'+new_name+'_ANNUITY(ENG)(CURR_APP)(BUREAU)'] = df[prefix_feature+'(BUREAU)']/df['AMT_ANNUITY(CURR_APP)']
            df['PCT_'+new_name+'_CREDIT(ENG)(CURR_APP)(BUREAU)'] = df[prefix_feature+'(BUREAU)']/df['AMT_CREDIT(CURR_APP)']

    for feature in ['AMT_CREDIT_PER_DAY_FORESEEN(ENG)', 'AMT_CREDIT_PER_DAY_FACT(ENG)',
                    'AMT_ANNUITY_PER_DAY_FORESEEN(ENG)', 'AMT_ANNUITY_PER_DAY_FACT(ENG)']:
        new_name = feature.split('(ENG)')[0]
        df['PCT_'+new_name+'_INCOME(ENG)(CURR_APP)(BUREAU)'] = df[feature+'(BUREAU)']/df['AMT_INCOME_TOTAL(CURR_APP)']
        df['PCT_'+new_name+'_ANNUITY(ENG)(CURR_APP)(BUREAU)'] = df[feature+'(BUREAU)']/df['AMT_ANNUITY(CURR_APP)']
        df['PCT_'+new_name+'_CREDIT(ENG)(CURR_APP)(BUREAU)'] = df[feature+'(BUREAU)']/df['AMT_CREDIT(CURR_APP)']

    for feature in ['AMT_APPLICATION', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'AMT_DOWN_PAYMENT',
                    'AMT_DENIED(ENG)', 'AMT_GAP_APPLICATION(ENG)', 'AMT_GAP_CREDIT(ENG)', 'AMT_CREDIT_PER_MONTH(ENG)',
                    'AMT_DENIED_PER_MONTH(ENG)', 'AMT_GAP_APPLICATION_PER_MONTH(ENG)', 'AMT_GAP_CREDIT_PER_MONTH(ENG)',
                    'ALL_AMT_CREDIT(ENG)', 'ALL_AMT_ANNUITY(ENG)', 'ALL_AMT_DOWN_PAYMENT(ENG)']:
        new_name = feature.split('(ENG)')[0]
        df['PCT_'+new_name+'_INCOME(ENG)(CURR_APP)(PREV_APP)'] = df[feature+'(PREV_APP)']/df['AMT_INCOME_TOTAL(CURR_APP)']
        df['PCT_'+new_name+'_ANNUITY(ENG)(CURR_APP)(PREV_APP)'] = df[feature+'(PREV_APP)']/df['AMT_ANNUITY(CURR_APP)']
        df['PCT_'+new_name+'_CREDIT(ENG)(CURR_APP)(PREV_APP)'] = df[feature+'(PREV_APP)']/df['AMT_CREDIT(CURR_APP)']

    df['AMT_CREDIT_PER_MONTH_DERIV(ENG)(CURR_APP)(PREV_APP)'] = df['AMT_CREDIT(CURR_APP)']/df['CNT_PAYMENT(PREV_APP)']
    df['PCT_CREDIT_PER_MONTH_DERIV_INCOME(ENG)(CURR_APP)(PREV_APP)'] = df['AMT_CREDIT_PER_MONTH_DERIV(ENG)(CURR_APP)(PREV_APP)']/df['AMT_INCOME_TOTAL(CURR_APP)']
    df['PCT_CREDIT_PER_MONTH_DERIV_ANNUITY(ENG)(CURR_APP)(PREV_APP)'] = df['AMT_CREDIT_PER_MONTH_DERIV(ENG)(CURR_APP)(PREV_APP)']/df['AMT_ANNUITY(CURR_APP)']

    df['RATE_ANNUITY(ENG)(BUREAU)(PREV_APP)'] = df['AMT_ANNUITY(BUREAU)']/df['AMT_ANNUITY(PREV_APP)']
    df['RATE_CREDIT(ENG)(BUREAU)(PREV_APP)'] = df['AMT_CREDIT_SUM(BUREAU)']/df['AMT_CREDIT(PREV_APP)']

    for feature in ['AMT_INSTALMENT', 'AMT_PAYMENT', 'AMT_INSTALMENT_DEBT(ENG)', 'MAXIMAL_AMT_INSTALMENT_DEBT(ENG)',
                    'MINIMAL_AMT_INSTALMENT_DEBT(ENG)', 'MONTHLY_AMT_INSTALMENT_SUM(ENG)', 'MONTHLY_AMT_PAYMENT_SUM(ENG)',
                    'AMT_PAYMENT', 'ALL_AMT_PAYMENT(ENG)', 'ALL_AMT_INSTALMENT_DEBT(ENG)']:
        new_name = feature.split('(ENG)')[0]
        df['PCT_'+new_name+'_INCOME(ENG)(CURR_APP)(INST_PAYM)'] = df[feature+'(INST_PAYM)']/df['AMT_INCOME_TOTAL(CURR_APP)']
        df['PCT_'+new_name+'_ANNUITY(ENG)(CURR_APP)(INST_PAYM)'] = df[feature+'(INST_PAYM)']/df['AMT_ANNUITY(CURR_APP)']
        df['PCT_'+new_name+'_CREDIT(ENG)(CURR_APP)(INST_PAYM)'] = df[feature+'(INST_PAYM)']/df['AMT_CREDIT(CURR_APP)']

    df['CNT_ACTIVE(ENG)(BUREAU)(POS_CASH)'] = df['CNT_ACTIVE(ENG)(BUREAU)'] + df['CNT_ACTIVE(ENG)(POS_CASH)']
    df['CNT_ACTIVE(ENG)(BUREAU)(CRED_CARD)'] = df['CNT_ACTIVE(ENG)(BUREAU)'] + df['CNT_ACTIVE(ENG)(CRED_CARD)']

    df['PCT_ALL_AMT_PAYMENT_CREDIT(ENG)(PREV_APP)(INST_PAYM)'] = df['ALL_AMT_PAYMENT(ENG)(INST_PAYM)']/df['ALL_AMT_CREDIT(ENG)(PREV_APP)']
    df['PCT_ALL_AMT_INSTALMENT_DEBT_CREDIT(ENG)(PREV_APP)(INST_PAYM)'] = df['ALL_AMT_INSTALMENT_DEBT(ENG)(INST_PAYM)']/df['ALL_AMT_CREDIT(ENG)(PREV_APP)']
    df['PCT_ALL_AMT_PAYMENT_ANNUITY(ENG)(PREV_APP)(INST_PAYM)'] = df['ALL_AMT_PAYMENT(ENG)(INST_PAYM)']/df['ALL_AMT_ANNUITY(ENG)(PREV_APP)']
    df['PCT_ALL_AMT_INSTALMENT_DEBT_ANNUITY(ENG)(PREV_APP)(INST_PAYM)'] = df['ALL_AMT_INSTALMENT_DEBT(ENG)(INST_PAYM)']/df['ALL_AMT_ANNUITY(ENG)(PREV_APP)']

    for feature in ['AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'MAXIMAL_AMT_BALANCE(ENG)', 'MAXIMAL_AMT_CREDIT_LIMIT_ACTUAL(ENG)',
                    'MINIMAL_AMT_BALANCE(ENG)', 'MINIMAL_AMT_CREDIT_LIMIT_ACTUAL(ENG)', 'MONTHLY_AMT_BALANCE_SUM(ENG)',
                    'MONTHLY_AMT_CREDIT_LIMIT_ACTUAL_SUM(ENG)']:
        new_name = feature.split('(ENG)')[0]
        df['PCT_'+new_name+'_INCOME(ENG)(CURR_APP)(CRED_CARD)'] = df[feature+'(CRED_CARD)']/df['AMT_INCOME_TOTAL(CURR_APP)']
        df['PCT_'+new_name+'_ANNUITY(ENG)(CURR_APP)(CRED_CARD)'] = df[feature+'(CRED_CARD)']/df['AMT_ANNUITY(CURR_APP)']
        df['PCT_'+new_name+'_CREDIT(ENG)(CURR_APP)(CRED_CARD)'] = df[feature+'(CRED_CARD)']/df['AMT_CREDIT(CURR_APP)']

start = time()

application = load(open('intermediary/application.pkl', 'rb'), encoding='latin1')

pkl_files = ['bureau', 'previous_application', 'installments_payments', 'credit_card_balance', 'pos_cash_balance']

for pkl_file in pkl_files:
    application = application.merge(load(open('intermediary/{}.pkl'.format(pkl_file), 'rb'), encoding='latin1'), on='SK_ID_CURR', how='left')

pos_merge_eng(application)

for column in application.columns:
    application[column].replace([np.inf, -np.inf], [np.nan, np.nan], inplace=True)
    if application[column].std()==0:
        application.drop(columns=[column], inplace=True)

correlations = application.corr().abs()

features = list(correlations.columns)
removed_features = []

for i in range(len(features)-1):
    for j in range(i+1, len(features)):
        i_feature = features[i]
        j_feature = features[j]

        if i_feature == j_feature or correlations.at[i_feature, j_feature] < 1:
            continue

        if tuple(application[i_feature].notna()) != tuple(application[j_feature].notna()):
            continue

        minn = min(i_feature, j_feature)
        maxx = max(i_feature, j_feature)

        if maxx not in removed_features and minn!=TARGET_COL:
            removed_features.append(minn)

application.drop(columns=removed_features, inplace=True)

correlations = application.corr().abs()
dump(correlations, open('intermediary/correlations.pkl', 'wb'))

del correlations
collect()

application = optimize(application)

train = application[application[TARGET_COL]!=2].reset_index(drop=True)
test = application[application[TARGET_COL]==2].drop(columns=[TARGET_COL]).reset_index(drop=True)

dump(train, open('intermediary/train.pkl', 'wb'))
dump(test, open('intermediary/test.pkl', 'wb'))

features = list(test.columns)
features.remove('SK_ID_CURR')

dump(features, open('intermediary/features.pkl', 'wb'))

print('compiled {} features ({:.2f}m)'.format(len(features), (time()-start)/60))
