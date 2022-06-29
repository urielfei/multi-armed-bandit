"""
This script creates 2 graphs
1. Mean reward by Round for algorithm LIN UCB
2. Mean reward by Arm for algorithm LIN UCB- Every arm starts from zero.
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

### Create Data for graphs
df = pd.read_csv('d.csv')
g = df.groupby('arm_LUCB',as_index=False).cumsum()
df['cum_reward_arm'] = g
df['cum_reward'] = df['reward_LUCB'].cumsum()
cum_reward_pct = []

for round in range(0, df.shape[0]):
    cd = df['cum_reward'][round]/(round+1)
    cum_reward_pct.append(cd)

df['cum_reward_pct'] = cum_reward_pct


cum_reward_arm_pct_lists = []
gr = df.groupby('arm_LUCB',as_index=False)
for ind,dat in gr:
    gr_data = dat.reset_index(drop=False)
    c_a = gr_data['cum_reward_arm']/(gr_data['index']+1)
    cum_reward_arm_pct_lists.append(c_a)

cum_reward_arm_pct = []
for sublist in cum_reward_arm_pct_lists:
    for item in sublist:
        cum_reward_arm_pct.append(item)

df['cum_reward_arm_pct'] = cum_reward_arm_pct

#### Mean Cumulative Reward
plt.plot(df.index,df['cum_reward_pct'])
plt.xlabel('Rounds')
plt.ylabel('Per-Round Cumulative Reward')
plt.title('Mean Reward by round')
plt.show()
plt.show()

###Graph mean Reward By Arm
gr = df.groupby('arm_LUCB',as_index=False)
fig, ax = plt.subplots(figsize=(15,7))
#dict_arm = {0:'red',1:'green',2:'yellow',3:'black',4:'blue'}
for ind,dat in gr:
    gr_data = dat[['arm_LUCB','cum_reward_arm_pct']].reset_index(drop=True)
    ax.plot(gr_data.index,gr_data['cum_reward_arm_pct'],label="Group"+str(ind)
            #,color=dict_arm)
            )

plt.xlabel('Rounds')
plt.ylabel('Per-Round Cumulative Reward')
plt.title('Mean Reward by Arm')
plt.legend()
plt.show()
