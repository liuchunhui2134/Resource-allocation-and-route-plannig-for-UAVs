import random
import numpy as np
import matplotlib.pyplot as plt
from Environment import *


class PSO:
    def __init__(self, parameters):

        self.NGEN = parameters[0]      # size of iterations
        self.pop_size = parameters[1]  # number of population

        self.UAV_num = parameters[2]   # number of UAVs

        self.fre_bound = parameters[3]  # frequence range

        self.power_bound = parameters[4]  # power range

        self.pop_x = np.zeros((self.pop_size, self.UAV_num, 3, 2)).astype(int)  # locations for particles
        self.pop_v = np.zeros((self.pop_size, self.UAV_num, 3, 2)).astype(int)  # velocity for particles
        self.p_best = np.zeros((self.pop_size, self.UAV_num, 3, 2)).astype(int)
        self.g_best = np.zeros((1, self.UAV_num, 3, 2)).astype(int)

        temp = -1
        for i in range(self.pop_size):
            for j in range(self.UAV_num):
                for k in range(3):
                    self.pop_x[i][j][k][0] = np.random.randint(self.fre_bound[0], self.fre_bound[1])
                    self.pop_x[i][j][k][1] = np.random.randint(self.power_bound[0], self.power_bound[1])
                    self.pop_v[i][j][k][0] = random.uniform(0, 1)
                    self.pop_v[i][j][k][1] = random.uniform(0, 1)
            self.p_best[i] = self.pop_x[i]
            fit = self.fitness(self.p_best[i])
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit

    def fitness(self, ind_var):
        rate = 0
        channel_select = ind_var[:, :, 0].astype(int)
        power_select = ind_var[:, :, 1].astype(int)
        actions = np.concatenate((channel_select[..., np.newaxis], power_select[..., np.newaxis]), axis=2)
        Env.activate_links = np.ones((self.UAV_num, 3), dtype='bool')
        for j in range(self.UAV_num):
            for k in range(3):
                U2I_pso, U2U_pso, time_left = Env.Compute_PSO_fitness(actions, [j, k])
                U2I_pso = U2I_pso.T.reshape([-1])
                U2U_pso = U2U_pso.T.reshape([-1])
                U2I_reward_pso = (U2I_pso[actions[j, k, 0] + 16 * actions[j, k, 1]] - np.min(U2I_pso)) / (np.max(U2I_pso) - np.min(U2I_pso) + 0.000001)
                U2U_reward_pso = (U2U_pso[actions[j, k, 0] + 16 * actions[j, k, 1]] - np.min(U2U_pso)) / (np.max(U2U_pso) - np.min(U2U_pso) + 0.000001)
                lambdda = 0.1
                t = lambdda * U2I_reward_pso + (1 - lambdda) * U2U_reward_pso
                rate += t - (Env.U2U_limit-time_left)/Env.U2U_limit
        rate = rate / (3*self.UAV_num)

        return rate

    def update_operator(self, pop_size):
        c1 = 2
        c2 = 2
        w = 0.4
        for i in range(pop_size):

            self.pop_v[i] = w * self.pop_v[i] + c1 * random.uniform(0, 1) * (self.p_best[i] - self.pop_x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.pop_x[i])

            for p in range(self.UAV_num):
                for q in range(3):
                    self.pop_x[i][p][q][0] = round(self.pop_x[i][p][q][0] + self.pop_v[i][p][q][0]) % 16
                    self.pop_x[i][p][q][1] = round(self.pop_x[i][p][q][1] + self.pop_v[i][p][q][1]) % 4

            if self.fitness(self.pop_x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.pop_x[i]
            if self.fitness(self.pop_x[i]) > self.fitness(self.g_best):
                self.g_best = self.pop_x[i]

    def main_pso(self):
        popobj = []
        self.ng_best = np.zeros((1, self.UAV_num, 3, 2))[0].astype(int)
        for gen in range(self.NGEN):
            self.update_operator(self.pop_size)
            if self.fitness(self.g_best) > self.fitness(self.ng_best):
                self.ng_best = self.g_best.copy()
            popobj.append(self.fitness(self.ng_best))

        plt.figure()
        plt.title("Figure1")
        plt.xlabel("iterators", size=14)
        plt.ylabel("fitness", size=14)
        t = [t for t in range(self.NGEN)]
        plt.plot(t, popobj, color='b', linewidth=2)
        plt.show()

        return self.ng_best


if __name__ == '__main__':
    NGEN = 100
    popsize = 96
    UAV_num = 32
    fre_bound = [0, 16]
    power_bound = [0, 4]
    Env = Environ()
    Env.new_random_game(UAV_num)
    parameters = [NGEN, popsize, UAV_num, fre_bound, power_bound]

    test_steps = 100
    U2Irate_pso = np.zeros(test_steps)
    U2Urate_pso = np.zeros(test_steps)
    success_pso = np.zeros(test_steps)
    for test_i in range(test_steps):
        pso = PSO(parameters)
        ng_best = pso.main_pso()
        print('Generation {}'.format(str(test_i + 1)))
        print('The resource allocation policy: {}'.format(ng_best))
        print('The maximum fitness value: {}'.format(pso.fitness(ng_best)))
        gen_U2Ireward, gen_U2Ureward, percent = Env.act_pso(ng_best)
        U2Irate_pso[test_i] = np.sum(gen_U2Ireward)
        U2Urate_pso[test_i] = np.sum(gen_U2Ureward)
        success_pso[test_i] = percent

    print('mean of U2I rate is that ', np.mean(U2Irate_pso))
    print('mean of U2U rate is that ', np.mean(U2Urate_pso))
    print('mean of percent is ', np.mean(success_pso))
