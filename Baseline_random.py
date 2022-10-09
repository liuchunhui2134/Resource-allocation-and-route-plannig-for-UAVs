from __future__ import division, print_function
import numpy as np 
from Environment import *
import matplotlib.pyplot as plt

# This py file using the random algorithm.


def main():
    n = 32
    Env = Environ()
    number_of_game = 10
    n_step = 200
    U2I_Rate_List = np.zeros([number_of_game, n_step])
    U2U_Rate_List = np.zeros([number_of_game, n_step])
    Success_Percent = np.zeros([number_of_game, n_step])
    for game_idx in range(number_of_game):
        print(game_idx)
        Env.new_random_game(n)
        for i in range(n_step):
            # print(i)
            actions = np.random.randint(0, 16, [n, 3])
            power_selection = np.random.randint(0, 4, actions.shape)
            actions = np.concatenate((actions[..., np.newaxis], power_selection[..., np.newaxis]), axis=2)
            U2Ireward, U2Ureward, percent = Env.act_random(actions)
            U2I_Rate_List[game_idx, i] = np.sum(U2Ireward)
            U2U_Rate_List[game_idx, i] = np.sum(U2Ureward)
            Success_Percent[game_idx, i] = percent
        # print(np.sum(reward))
        # print('percentage here is ', percent)
        print('The number of UAVs is ', n)
        print('mean of U2I rate is that ', np.mean(U2I_Rate_List))
        print('mean of U2U rate is that ', np.mean(U2U_Rate_List))
        print('mean of percent is ', np.mean(Success_Percent[:, -1]))


main()
