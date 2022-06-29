"""
Script that shuffles the users' order to entry the algorithm
Goal: Ensure that the order doesn't affect the results
"""
import numpy as np
import pandas as pd
from run_file import LinUCB
import time
from collections import Counter
import random
arms = np.load('arms.npy')
rewards = np.load('rewards.npy')
contexts = np.load('contexts.npy')
num_of_events = np.load('num_of_events.npy')

#b number of simulations
b = 1

def sim_shuffle():
    d = list(range(num_of_events))
    shuffled_range = random.sample(d, len(d))
    return shuffled_range


def offlineShuffleEvaluate(mab, arms, rewards, contexts, n_rounds=None):
    h0 = []  # Arms History list
    R0 = []  # Total Reward - If action = play() then we check the reward, either reward=0

    count = 0
    for event in sim_shuffle():
        print(event)

        if len(h0) == n_rounds:  # If reach required number of rounds then stop
            break

        # Play an arm
        action = mab.play(t_round=len(h0)+1, context=contexts[event])
        print(action)
        print(arms[event])
        if action == arms[event]:
            count += 1
            h0.append(action)
            R0.append(rewards[event])
            mab.update(arms[event], rewards[event], contexts[event])

        #else:

            #count += 1
            #h0.append(action)
            #Change here if needed
            #R0.append(0) #R0.append(rewards[event])
            #mab.update(arms[event], rewards[event], contexts[event])


    return R0,h0


if __name__ == '__main__':
    start_time = time.time()
    results = []
    results_arms = []

    for ind in range(b):
        mab_linUCB = LinUCB(5, 30, 0.5)
        results_LinUCB, arms_chosen_LinUCB = offlineShuffleEvaluate(mab_linUCB, arms, rewards, contexts, 'LinUCB')
        results.append(results_LinUCB)
        results_arms.append(arms_chosen_LinUCB)
        print(ind)
        print('LinUCB average reward', "%.4f" % np.mean(results_LinUCB))
        print("--- %.4f seconds ---" % (time.time() - start_time))
        c = Counter(arms_chosen_LinUCB)
        print('Arms',c)

print("--- %.4f seconds ---" % (time.time() - start_time))



mean_results = [np.mean(sublist) for sublist in results]
print(mean_results)
print(np.mean(mean_results))
print(np.var(mean_results))