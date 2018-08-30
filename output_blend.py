from data_aux_functions import *
import matplotlib.pyplot as plt
from parameters import *
from pickle import load
from os import listdir
import seaborn as sns
import pandas as pd
import numpy as np
import sys

plt.switch_backend('agg')

count = 0
blend = None
timestamp_min = np.inf
timestamp_max = -np.inf

for filename in listdir('output/pre_sub'):
    if filename=='.gitkeep' or filename=='.DS_Store':
        continue
    df = load(open('output/pre_sub/'+filename, 'rb'), encoding='latin1').rename(columns={TARGET_COL:''})
    timestamp = int(filename.split('.')[0])
    timestamp_min = min(timestamp_min, timestamp)
    timestamp_max = max(timestamp_max, timestamp)
    if blend is None:
        blend = df
    else:
        blend = pd.concat([blend, df[['']]], axis=1)
    count += 1

timestamp_min = timestamp_min - (timestamp_max-timestamp_min)/count
delta = round((timestamp_max-timestamp_min)/3600, 2)
print('Training time: {}h'.format(delta))

data = blend.drop(columns=[ID_COL]).values
blend[TARGET_COL] = get_blend(data)
blend[[ID_COL, TARGET_COL]].to_csv('output/blend.csv', index=False)

if count==1:
    exit()

n_pre_subs_range = range(max(2, count-100), count+1)
stds_sums = []

for n_pre_subs in n_pre_subs_range:
    stds_sums.append(data[:, 0:n_pre_subs].std(axis=1).sum())

plt.plot(n_pre_subs_range, stds_sums)
plt.text(n_pre_subs_range[-1], stds_sums[-1], stds_sums[-1])
plt.xlabel('# pre-subs')
plt.ylabel('sum of predictions\' stds')
plt.title('uncertainty over time')
plt.savefig('output/stds_vs_npresubs.png')

blend['STD'] = data.std(axis=1)

sns.jointplot(x=blend[TARGET_COL], y=blend['STD'], kind='reg', size=10)
plt.title('stds vs target')
plt.savefig('output/stds_vs_target.png')
