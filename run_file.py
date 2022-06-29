import numpy as np
import pandas as pd
from numpy.linalg import inv
import warnings
warnings.filterwarnings('ignore')
import random
from collections import Counter
import matplotlib.pyplot as plt
import time

arms = np.load('arms.npy')
rewards = np.load('rewards.npy')
contexts = np.load('contexts.npy')
num_of_events = np.load('num_of_events.npy')



class LinUCB():
    def __init__(self, n_arms, n_dims, alpha):
        self.n_arms = n_arms
        self.n_dims = n_dims
        self.alpha = alpha

        self.A = {}  # D.T*D + I for each arm
        self.b = {}  # rewards

        # step 1: param initialization
        # For each arm, create A, b
        for arm in range(1, self.n_arms+1):
            if arm not in self.A:
                self.A[arm] = np.identity(n_dims)
            if arm not in self.b:
                self.b[arm] = np.zeros(n_dims)

    # step 2: compute UCB for each Arm
    def play(self, context, t_round=None):
        p_t = {}

        for arm in range(1, self.n_arms + 1):
            theta_a = np.dot(inv(self.A[arm]), self.b[arm])
            std = np.sqrt(np.linalg.multi_dot([np.transpose(context), np.linalg.inv(self.A[arm]), context]))
            p_ta = np.dot(theta_a.T, context) + self.alpha * std
            if not np.isnan(p_ta):  # make sure the result of calculation is valid number
                p_t[arm] = p_ta

        # step 3: take action
        max_UCB = max(p_t.values())
        print(max_UCB)
        max_UCB_key = [key for key, value in p_t.items() if value == max_UCB]
        if len(max_UCB_key) > 1:
            action = np.random.choice(max_UCB_key)  # Tie Breaker
        else:
            action = max_UCB_key[0]
        return action

    # step 4: update
    def update(self, arm, reward, context):
        self.A[arm] = np.add(self.A[arm], np.dot(context, np.transpose(context)))
        self.b[arm] = np.add(self.b[arm], np.dot(reward, context))

    def update_alpha(self,alpha_max):
        self.alpha = alpha_max


class UCB():
    def __init__(self, n_arms, rho, Q_0 = np.inf):
        self.n_arms = n_arms
        self.rho = rho
        self.Q_0 = Q_0
        self.arm_visit_count = {}
        self.arm_total_reward = {}

        self.arm_with_avg_reward = {}
        for arm in range(1, self.n_arms + 1):
            self.arm_with_avg_reward[arm] = self.Q_0  # Initial all the arm with Q0

            self.arm_visit_count[arm] = 0  # Initial all the arm with zero number of visits
            self.arm_total_reward[arm] = 0  # Initial all the arm with zero reward

    def play(self, t_round, context=None):
        temp_arm_with_Q = self.arm_with_avg_reward

        for arm in temp_arm_with_Q:
            if self.arm_visit_count[arm] == 0:  # Use Q0 for the first round
                continue


            else:
                # At t_round, calculate Q with exlpore boost for each arm
                explore_boost_const = self.rho * np.log(t_round) / self.arm_visit_count[arm]

                temp_arm_with_Q[arm] = temp_arm_with_Q[arm] + np.sqrt(explore_boost_const)

        # Getting the highest value from Q, then find the corresponding key and append them
        highest = max(temp_arm_with_Q.values())
        highest_Qs = [key for key, value in temp_arm_with_Q.items() if value == highest]
        if len(highest_Qs) > 1:
            action = np.random.choice(highest_Qs)  # Tie Breaker
        else:
            action = highest_Qs[0]
        return action

    def update(self, arm, reward, context=True):
        self.arm_visit_count[arm] += 1
        self.arm_total_reward[arm] += reward
        updated_reward = self.arm_total_reward[arm] / self.arm_visit_count[arm]

        self.arm_with_avg_reward.update({arm: updated_reward})

        return self.arm_with_avg_reward


class LinThompson():
    def __init__(self, n_arms, n_dims, v):
        self.n_arms = n_arms
        self.n_dims = n_dims
        self.v = v

        self.B = np.identity(self.n_dims)  # Initial B with identity matrix which has ndims dimension
        self.f = np.zeros(self.n_dims)  # Initial total payoff with ndims of zeros
        self.u = np.zeros(self.n_dims)  # Initial parameter mu with ndims of zeros

    def play(self, t_round, context):
        arm_with_Q = {}

        # Calculate prior from multivariate Gaussian distribution
        u_t = np.random.multivariate_normal(self.u, self.v * self.v * np.linalg.inv(self.B))

        for arm in range(1, self.n_arms + 1):
            # calculate posterior distribution for each arm
            arm_with_Q[arm] = np.dot(np.transpose(context), u_t)

        # Getting the highest value from posterior distribution, then find the corresponding key and append them
        highest = max(arm_with_Q.values())
        highest_Qs = [key for key, value in arm_with_Q.items() if value == highest]
        if len(highest_Qs) > 1:
            action = np.random.choice(highest_Qs)  # Tie Breaker
        else:
            action = highest_Qs[0]
        return action

    def update(self, arm, reward, context):
        assert (arm > 0 and arm <= self.n_arms), "arm must be positive integers and no larger than self.narms"
        assert (type(reward) == float or type(reward) == np.float64), "reward must be floating point"


        if arm <= self.n_arms:
            # Reshap the vector to matrix, or the calculation will be incorrect
            # because the transpose will not take effects
            context_times_contextT = np.dot(context, context.T)
            # Update B
            self.B = np.add(self.B, context_times_contextT)
            # Update reward f
            self.f = np.add(self.f, np.multiply(reward, context))
            # Update mu
            self.u = np.dot(np.linalg.inv(self.B), self.f)

def tune_alpha(num_round_now,by=10):
    interval = np.linspace(0.001,1,by)
    results_LinUCB_with_alpha = []
    for alpha in interval:
        mab = LinUCB(5, 30, alpha)
        results_LinUCB = offlineEvaluate(mab, arms, rewards, contexts,num_of_events,n_rounds=num_round_now)[0] #Check the best alpha based on the sample we had
        results_LinUCB_with_alpha.append(np.mean(results_LinUCB))

    ind = results_LinUCB_with_alpha.index(max(results_LinUCB_with_alpha))
    alpha_max = interval[ind]
    print('Updating Alpha')
    print(alpha_max)
    return alpha_max


def offlineEvaluate(mab, arms, rewards, contexts, num_of_events,n_rounds=None):
    h0 = []  # Arms History list
    R0 = []  # Total Reward - If action = play() then we check the reward, either reward=0

    count = 0
    for event in range(num_of_events):

        if event == n_rounds:  # If reach required number of rounds then stop
            break

        # if event % 2000 == 0 and event!= 0 and event<=10000:  # Update alpha every 2000 rounds till 10k
        #     print(event)
        #     mab.update_alpha(tune_alpha(event))

        # Play an arm
        action = mab.play(t_round=len(h0)+1, context=contexts[event])
        if action == arms[event]:
            count += 1
            h0.append(action)
            R0.append(rewards[event])
            mab.update(arms[event], rewards[event], contexts[event])

        else:

            #count += 1
            h0.append(action)
            #Change here if needed
            #R0.append(0) #R0.append(rewards[event])
            #mab.update(arms[event], rewards[event], contexts[event])


    return R0,h0



if __name__ == '__main__':
    start_time = time.time()

    mab_linUCB = LinUCB(5, 30, 0.01)
    results_LinUCB, arms_chosen_LinUCB = offlineEvaluate(mab_linUCB, arms, rewards, contexts, num_of_events, 'LinUCB')
    print('LinUCB average reward', "%.4f" % np.mean(results_LinUCB))
    print("--- %.4f seconds ---" % (time.time() - start_time))

    # mab_LinThompson = LinThompson(5, 30, 0.07)
    # results_LinThompson, arms_chosen_LinThompson = offlineEvaluate(mab_LinThompson, arms, rewards, contexts)
    # print('LinThompson average reward', "%.4f" % np.mean(results_LinThompson))
    # print("--- %.4f seconds ---" % (time.time() - start_time))
    #
    #
    # mab_UCB = UCB(5, 1.)
    # results_UCB, arms_chosen_UCB = offlineEvaluate(mab_UCB, arms, rewards, contexts)
    # print('UCB average reward', "%.4f" % np.mean(results_UCB))
    # print("--- %.4f seconds ---" % (time.time() - start_time))

    c = Counter(arms_chosen_LinUCB)
    print('Arms',c)

    dict_linUCB = {'arm_LUCB':arms_chosen_LinUCB, 'reward_LUCB': results_LinUCB}
    # dict_UCB = {'arm_UCB': arms_chosen_UCB, 'reward_UCB': results_UCB}
    # dict_linTom = {'arm_LTMS': arms_chosen_LinThompson, 'reward_LTMS': results_LinThompson}

    df_linUCB = pd.DataFrame(dict_linUCB)
    df_linUCB.to_csv('d.csv', index=False)

    # df_UCB = pd.DataFrame(dict_UCB)
    # df_UCB.to_csv('e.csv', index=False)
    #
    # df_linTOM = pd.DataFrame(dict_linTom)
    # df_linTOM.to_csv('f.csv', index=False)
