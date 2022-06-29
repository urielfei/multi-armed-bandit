"""
This script creates 2 graphs
1. Mean reward by Algorithm
2. Mean reward by Arm for algorithm LIN UCB- Every arm starts from zero.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

### Create Data for graphs
df_LinUCB = pd.read_csv('d.csv')
df_UCB = pd.read_csv('e.csv')
df_LTMS = pd.read_csv('f.csv')


cum_reward_LUCB = []
for i in range(df_LinUCB.shape[0]):
    cum_reward_LUCB.append(np.mean(df_LinUCB['reward_LUCB'][0:i+1]))

cum_reward_UCB = []
for i in range(df_UCB.shape[0]):
    cum_reward_UCB.append(np.mean(df_UCB['reward_UCB'][0:i+1]))

cum_reward_LTMS = []
for i in range(df_LTMS.shape[0]):
    cum_reward_LTMS.append(np.mean(df_LTMS['reward_LTMS'][0:i+1]))


print(len(cum_reward_LUCB))
print(df_LinUCB.shape)


plt.plot(df_LinUCB.index,cum_reward_LUCB, label='LinUCB')
plt.plot(df_UCB.index,cum_reward_UCB, label='UCB')
plt.plot(df_LTMS.index,cum_reward_LTMS, label='LinThompson')
plt.xlabel('Rounds')
plt.ylabel('Per-Round Cumulative Reward')
plt.title('Mean Reward by round')
plt.legend()

plt.show()
