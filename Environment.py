from __future__ import division
import numpy as np
import time
import random
import math
import pandas as pd
import sys
from PIL import Image
from matplotlib import pyplot as plt


# This file is revised for more precise and concise expression.


class U2Uchannels:
    # Simulator of the U2U Channels
    def __init__(self, n_UAV, n_RB):
        self.t = 0
        self.h_bs = 200
        self.h_ms = 200
        self.fc = 0.7
        self.decorrelation_distance = 50
        self.shadow_std = 3
        self.n_UAV = n_UAV
        self.n_RB = n_RB
        self.update_shadow([])

    def update_positions(self, positions):
        self.positions = positions

    def update_pathloss(self):
        self.PathLoss = np.zeros(shape=(len(self.positions), len(self.positions)))
        for i in range(len(self.positions)):
            for j in range(len(self.positions)):
                self.PathLoss[i][j] = self.get_path_loss(self.positions[i], self.positions[j])

    def update_shadow(self, delta_distance_list):
        delta_distance = np.zeros((len(delta_distance_list), len(delta_distance_list)))
        for i in range(len(delta_distance)):
            for j in range(len(delta_distance)):
                delta_distance[i][j] = delta_distance_list[i] + delta_distance_list[j]
        if len(delta_distance_list) == 0:
            self.Shadow = np.random.normal(0, self.shadow_std, size=(self.n_UAV, self.n_UAV))
        else:
            self.Shadow = np.exp(-1*(delta_distance/self.decorrelation_distance)) * self.Shadow +\
                         np.sqrt(1 - np.exp(-2*(delta_distance/self.decorrelation_distance))) * np.random.normal(0, self.shadow_std, size=(self.n_UAV, self.n_UAV))

    def update_fast_fading(self):
        omega = 1
        k = 12
        h_rayleigh = np.sqrt(omega/2) * (np.random.normal(size=(self.n_UAV, self.n_UAV, self.n_RB)) + 1j * np.random.normal(size=(self.n_UAV, self.n_UAV, self.n_RB)))
        h_rician = np.sqrt(k / (k+1)) + np.sqrt(1 / (k+1)) * h_rayleigh
        self.FastFading = 20 * np.log10(np.abs(h_rician))

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1, d2) + 0.001
        PL = 32.44 + 20*np.log10(d) + 20*np.log10(self.fc)
        return PL


class U2Ichannels:
    # Simulator of the U2I channels
    def __init__(self, n_UAV, n_RB):
        self.h_bs = 35
        self.h_ms = 200
        self.fc = 0.7
        self.PLos = 0
        self.PNLos = 0
        self.Decorrelation_distance = 40
        self.BS_position = [12, 12]    # Suppose the BS is in the center
        self.shadow_std = 8
        self.n_UAV = n_UAV
        self.n_RB = n_RB
        self.update_shadow([])

    def update_positions(self, positions):
        self.positions = positions

    def update_pathloss(self):
        self.PathLoss = np.zeros(len(self.positions))
        for i in range(len(self.positions)):
            d1 = abs(self.positions[i][0] - self.BS_position[0])
            d2 = abs(self.positions[i][1] - self.BS_position[1])
            distance_2d = math.hypot(d1, d2)
            distance_3d = math.sqrt(distance_2d ** 2 + ((self.h_bs - self.h_ms) ** 2))
            self.PathLoss[i] = max(23.9-1.8*np.log10(self.h_ms), 20)*np.log10(distance_3d) + 20*np.log10(40*np.pi*self.fc/3)

    def update_shadow(self, delta_distance_list):
        shadow_std_LoS = 4.2 * np.exp((-1)*0.0046*self.h_ms)
        if len(delta_distance_list) == 0:
            self.Shadow_Los = np.random.normal(0, shadow_std_LoS, self.n_UAV)
        else:
            delta_distance = np.asarray(delta_distance_list)
            self.Shadow_Los = np.exp(-1*(delta_distance/self.Decorrelation_distance)) * self.Shadow_Los + np.sqrt(1-np.exp(-2*(delta_distance/self.Decorrelation_distance)))*np.random.normal(0, shadow_std_LoS, self.n_UAV)

    def update_fast_fading(self):
        A = 22.5 * np.log10(self.h_ms) - 4.72
        omega = 6.988 * np.exp(0.01659*self.h_ms)
        k = A**2 / (2*(omega**2))
        h_rayleigh = np.sqrt(omega/2) * (np.random.normal(size=(self.n_UAV, self.n_RB)) + 1j * np.random.normal(size=(self.n_UAV, self.n_RB)))
        h_rician = np.sqrt(k / (k+1)) + np.sqrt(1 / (k+1)) * h_rayleigh
        self.FastFading = 20 * np.log10(np.abs(h_rician))


class UAV:
    def __init__(self, start_position, p, p_next, velocity, destination, trajectory):
        self.position = start_position
        self.p = p
        self.p_next = p_next
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []
        self.destination = destination
        self.trajectory = trajectory


class Block:
    def __init__(self, position):
        self.position = position


class Address:
    def __init__(self, position):
        self.position = position


# Filter duplicate nodes in a path
class Filter:
    def __init__(self):
        self.b = 1

    def function(self, a):
        for i in a:
            a = a[a.index(i) + 1:]
            if i in a:
                return i, 1
            else:
                pass
        return 0, 0

    def filter(self, a):
        if a == []:
            return a
        while self.b == 1:
            (i, self.b) = self.function(a)
            c = [j for j, x in enumerate(a) if x == i]
            if c == []:
                return a
            else:
                a = a[0:c[0]] + a[c[-1]:]
        return a


filter = Filter()


# Generate feasible paths for UAVs
class FeasibleSolution:
    def __init__(self, map_data, destination_i):
        row = len(map_data)
        col = len(map_data)
        self.mapdata = []
        self.p_start = 0
        self.p_end = destination_i[1] + destination_i[0]*col
        if self.p_end != self.p_start:
            self.xs = self.p_start // col
            self.ys = self.p_start % col
            self.xe = self.p_end // col
            self.ye = self.p_end % col

            self.mapdata = map_data

            self.can = []
            self.popu = []
            self.end_popu = []

    def feasiblesolution_init(self, row, col):
        temp = []
        while temp == []:
            self.end_popu = []
            for xk in range(0, row):
                self.can = []
                for yk in range(0, col):
                    num = yk + xk * col
                    if self.mapdata[xk][yk] == 10:
                        self.can.append(num)
                length = len(self.can)
                a = (self.can[np.random.randint(0, length - 1)])
                self.end_popu.append(a)
            self.end_popu[0] = self.p_start
            self.end_popu[-1] = self.p_end
            temp = self.Generate_Continuous_Path(self.end_popu, row, col)
        self.end_popu = filter.filter(temp)
        return self.end_popu

    # Generate a new path
    def get_newsolution(self, oldsolution):
        row = 25
        col = 25
        temp = []
        while (temp == []):
            p_temp = []
            single_popu = oldsolution
            col = len(single_popu)
            if col > 3:
                first = np.random.randint(1, col - 2)
                p_temp = [p for p in range(1, col - 1)]
                p_temp.remove(first)
                second = np.random.choice(p_temp)
                if first != second:
                    if first < second:
                        single_popu = single_popu[0:first] + single_popu[second + 1:]
                    else:
                        single_popu = single_popu[0:second] + single_popu[first + 1:]
            else:
                if col == 3:
                    single_popu = single_popu[:1] + single_popu[2:]
                else:
                    return oldsolution
            temp = self.Generate_Continuous_Path(single_popu, 25, 25)
        new_popu = filter.filter(temp)
        return new_popu

    def Generate_Continuous_Path(self, old_popu, row, col):

        num_insert = 0
        self.new_popu = old_popu

        self.flag = 0
        self.length = len(self.new_popu)
        i = 0
        while i != self.length-1:

            x_now = (self.new_popu[i]) // col
            y_now = (self.new_popu[i]) % col
            x_next = (self.new_popu[i+1]) // col
            y_next = (self.new_popu[i+1]) % col

            max_iteration = 0

            while max(abs(x_next - x_now), abs(y_next - y_now)) != 1:

                x_insert = (x_next + x_now) // 2
                y_insert = (y_next + y_now) // 2

                flag1 = 0

                if self.mapdata[x_insert][y_insert] == 10:

                    num_insert = y_insert + x_insert * col

                    self.new_popu.insert(i+1, num_insert)

                else:
                    if (x_insert + 1 < row) and flag1 == 0:
                        if self.mapdata[x_insert+1][y_insert] == 10 and (y_insert) + (x_insert+1) * col not in self.new_popu:
                            num_insert = y_insert + (x_insert+1) * col
                            self.new_popu.insert(i + 1, num_insert)
                            flag1 = 1

                    if (x_insert + 1 < row) and (y_insert + 1 < col) and flag1 == 0:
                        if self.mapdata[x_insert+1][y_insert+1] == 10 and (y_insert+1) + (x_insert+1) * col not in self.new_popu:
                            num_insert = (y_insert+1) + (x_insert+1) * col
                            self.new_popu.insert(i + 1, num_insert)
                            flag1 = 1

                    if (y_insert + 1 < col) and flag1 == 0:
                        if self.mapdata[x_insert][y_insert+1] == 10 and (y_insert+1) + (x_insert) * col not in self.new_popu:
                            num_insert = (y_insert+1) + (x_insert) * col
                            self.new_popu.insert(i + 1, num_insert)
                            flag1 = 1

                    if (y_insert + 1 < col) and (x_insert - 1 >= 0) and flag1 == 0:
                        if self.mapdata[x_insert-1][y_insert+1] == 10 and (y_insert+1) + (x_insert-1) * col not in self.new_popu:
                            num_insert = (y_insert+1) + (x_insert-1) * col
                            self.new_popu.insert(i + 1, num_insert)
                            flag1 = 1

                    if (x_insert - 1 >= 0) and flag1 == 0:
                        if self.mapdata[x_insert-1][y_insert] == 10 and (y_insert) + (x_insert-1) * col not in self.new_popu:
                            num_insert = (y_insert) + (x_insert-1) * col
                            self.new_popu.insert(i + 1, num_insert)
                            flag1 = 1

                    if (x_insert - 1 >= 0) and (y_insert - 1 >= 0) and flag1 == 0:
                        if self.mapdata[x_insert-1][y_insert-1] == 10 and (y_insert-1) + (x_insert-1) * col not in self.new_popu:
                            num_insert = (y_insert-1) + (x_insert-1) * col
                            self.new_popu.insert(i + 1, num_insert)
                            flag1 = 1

                    if (y_insert - 1 >= 0) and flag1 == 0:
                        if self.mapdata[x_insert][y_insert-1] == 10 and (y_insert-1) + (x_insert) * col not in self.new_popu:
                            num_insert = (y_insert-1) + (x_insert) * col
                            self.new_popu.insert(i + 1, num_insert)
                            flag1 = 1

                    if (y_insert - 1 >= 0) and (x_insert + 1 < row) and flag1 == 0:
                        if self.mapdata[x_insert+1][y_insert-1] == 10 and (y_insert-1) + (x_insert+1) * col not in self.new_popu:
                            num_insert = (y_insert-1) + (x_insert+1) * col
                            self.new_popu.insert(i + 1, num_insert)
                            flag1 = 1

                    if flag1 == 0:
                        x_insert = ((x_insert + x_now) // 2)
                        y_insert = ((y_insert + y_now) // 2)
                        if self.mapdata[x_insert][y_insert] == 10:
                            num_insert = y_insert + x_insert * col
                        else:
                            node = []
                            for y_n in range(0, col):
                                if self.mapdata[x_insert][y_n] == 10:
                                    node.append(y_n + x_insert*col)
                            if len(node) != 0:
                                num_insert = np.random.choice(node)

                x_next = num_insert // col
                y_next = num_insert % col


                max_iteration += 1
                if max_iteration > 50:
                    self.new_popu = []
                    break
            if self.new_popu == []:
                break

            self.length = len(self.new_popu)
            i = i + 1
        return self.new_popu

    # Calculate the fitness of path
    def calvalue(self, popu, col):

        value_length = 0
        value_smooth = 0
        single_popu = popu
        single_length = len(single_popu)
        for j in range(single_length-1):
            x_now = single_popu[j] // col
            y_now = single_popu[j] % col
            x_next = single_popu[j + 1] // col
            y_next = single_popu[j + 1] % col
            if abs(x_now - x_next) + abs(y_now - y_next) == 1:
                value_length = value_length + 1
            elif max(abs(x_now - x_next), abs(y_now - y_next)) >= 2:
                value_length = value_length + 100
            else:
                value_length = value_length + 1.4

        for k in range(single_length-2):
            x_now = single_popu[k] // col
            y_now = single_popu[k] % col
            x_next = single_popu[k + 2] // col
            y_next = single_popu[k + 2] % col
            if abs(x_now - x_next) + abs(y_now - y_next) == 1:
                value_smooth = value_smooth + 100
            elif math.hypot(x_now - x_next, y_now - y_next) == 2 or math.hypot(x_now - x_next, y_now - y_next) == 2*math.sqrt(2):
                value_smooth = value_smooth + 1.4
            else:
                value_smooth = value_smooth + 1

        return 2*value_length + value_smooth


class Environ:
    # Environment Simulator: Provide states and rewards to agents.
    # Evolve to new state based on the actions taken by UAVs.
    def __init__(self):
        self.UAVs = []
        self.blocks = []
        self.addresses = []
        self.demands = []
        self.U2U_power_dB = 23  # dBm
        self.U2I_power_dB = 23  # dBm
        self.U2U_power_dB_List = [23, 12, 5, 1]             # the power levels
        # self.V2V_power = 10**(self.V2V_power_dB)
        # self.V2I_power = 10**(self.V2I_power_dB)
        self.sig2_dB = -114
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.uavAntGain = 8
        self.uavNoiseFigure = 9
        self.sig2 = 10**(self.sig2_dB/10) 
        self.U2U_Shadowing = []
        self.U2I_Shadowing = []
        self.n_RB = 16
        self.n_UAV = 32
        self.U2Uchannels = U2Uchannels(self.n_UAV, self.n_RB)
        self.U2Ichannels = U2Ichannels(self.n_UAV, self.n_RB)

        self.U2U_Interference_all = np.zeros((self.n_UAV, 3, self.n_RB)) + self.sig2
        self.n_step = 0

    def add_blocks(self, position):
        self.blocks.append(Block(position))

    def add_addresses(self, position):
        self.addresses.append(Address(position))

    def add_new_UAVs(self, start_position, p, p_next, start_velocity, destination, trajectory):
        self.UAVs.append(UAV(start_position, p, p_next, start_velocity, destination, trajectory))

    def route_planning(self, map_data, destination_i):
        T0 = 10
        T = T0
        maxgen = 100
        Lk = 100
        alfa = 0.95
        starttime = time.time()
        x_plot = []
        y_plot = []
        value_loss = []
        # print('Start time: ', starttime)
        feasiblesolution = FeasibleSolution(map_data, destination_i)
        oldsolution = feasiblesolution.feasiblesolution_init(25, 25)
        # print('Initial path', oldsolution)
        old_value = feasiblesolution.calvalue(oldsolution, 25)
        # print('The length of initial path', old_value)
        for i in range(maxgen):
            for j in range(Lk):
                # step2 Calculate the fitness value of the current path
                old_value = feasiblesolution.calvalue(oldsolution, 25)
                value_loss.append(old_value)
                # print('old', old_value)
                # step3 Generate a new path
                newsolution = feasiblesolution.get_newsolution(oldsolution)
                new_value = feasiblesolution.calvalue(newsolution, 25)
                # print('new', new_value)
                if new_value <= old_value:
                    oldsolution = newsolution
                else:
                    p = np.exp(-(new_value - old_value) / T)
                    if np.random.random() < p:
                        oldsolution = newsolution
            T = T * alfa
        # print('The path for UAV: ', oldsolution)
        old_value = feasiblesolution.calvalue(oldsolution, 25)
        # print('The length of path: ', old_value)
        stoptime = time.time()  # 结束时间
        # print('Stop time: ', stoptime)
        # print('Time for route planning: ', stoptime - starttime, '秒')
        for i in oldsolution:
            x = i // 25
            x_plot.append(x)
            y = i % 25
            y_plot.append(y)

        for i in range(1, len(x_plot) - 1):
            feasiblesolution.mapdata[x_plot[i]][y_plot[i]] = 10
        feasiblesolution.mapdata[x_plot[0]][y_plot[0]] = 2
        feasiblesolution.mapdata[x_plot[-1]][y_plot[-1]] = 8

        '''
        plt.figure()
        plt.imshow(feasiblesolution.mapdata, cmap=plt.cm.hot, interpolation='nearest', vmin=0, vmax=10)
        plt.plot(y_plot, x_plot)
        plt.xlim(-1, 25)  # 设置x轴范围
        plt.ylim(-1, 25)  # 设置y轴范围
        my_x_ticks = np.arange(0, 25, 1)
        my_y_ticks = np.arange(0, 25, 1)
        plt.xticks(my_x_ticks)
        plt.yticks(my_y_ticks)
        plt.grid(True)

        plt.figure()
        loss_x = [c for c in range(len(value_loss))]
        plt.plot(loss_x, value_loss)
        plt.show()
        '''

        return oldsolution, feasiblesolution.mapdata

    def add_new_UAVs_by_number(self, n):
        delta_time = 20
        map_data = [[10 for r in range(25)] for c in range(25)]
        obstacles = 25
        destinations = 32

        obstacles_list = [d for d in range(1, 25*25)]
        for i in range(0, obstacles):
            obstacle_num = np.random.choice(obstacles_list)
            obstacles_list.remove(obstacle_num)
            position = [obstacle_num // 25, obstacle_num % 25]
            map_data[position[0]][position[1]] = 0
            self.add_blocks(position)

        destinations_list = obstacles_list
        for j in range(0, destinations):
            destination_num = np.random.choice(destinations_list)
            destinations_list.remove(destination_num)
            position = [destination_num // 25, destination_num % 25]
            map_data[position[0]][position[1]] = 8
            self.add_addresses(position)

        start_position = [0, 0]
        map_data[start_position[0]][start_position[1]] = 2
        for k in range(0, n):
            p = 0
            p_next = 1
            start_velocity = np.random.randint(100, 200)
            destination = [self.addresses[k].position[0], self.addresses[k].position[1]]
            trajectory_i, map_data_i = self.route_planning(map_data, destination)
            self.add_new_UAVs(start_position, p, p_next, start_velocity, destination, trajectory_i)

        self.U2U_Shadowing = np.random.normal(0, 3, [len(self.UAVs), len(self.UAVs)])
        self.U2I_Shadowing = np.random.normal(0, 8, len(self.UAVs))
        self.delta_distance = np.asarray([c.velocity for c in self.UAVs])

        return map_data

    def renew_positions(self):
        for i in range(len(self.UAVs)):

            if self.UAVs[i].p_next > self.UAVs[i].p:
                if self.UAVs[i].p_next != len(self.UAVs[i].trajectory):
                    self.UAVs[i].position = [self.UAVs[i].trajectory[self.UAVs[i].p_next] // 25, self.UAVs[i].trajectory[self.UAVs[i].p_next] % 25]
                    self.UAVs[i].p = self.UAVs[i].p + 1
                    self.UAVs[i].p_next = self.UAVs[i].p_next + 1
                else:
                    self.UAVs[i].p = len(self.UAVs[i].trajectory) - 1
                    self.UAVs[i].p_next = self.UAVs[i].p - 1
                    self.UAVs[i].position = [self.UAVs[i].trajectory[self.UAVs[i].p_next] // 25, self.UAVs[i].trajectory[self.UAVs[i].p_next] % 25]
                    self.UAVs[i].p = self.UAVs[i].p - 1
                    self.UAVs[i].p_next = self.UAVs[i].p_next - 1
            else:
                if self.UAVs[i].p > self.UAVs[i].p_next:
                    if self.UAVs[i].p_next != -1:
                        self.UAVs[i].position = [self.UAVs[i].trajectory[self.UAVs[i].p_next] // 25, self.UAVs[i].trajectory[self.UAVs[i].p_next] % 25]
                        self.UAVs[i].p = self.UAVs[i].p - 1
                        self.UAVs[i].p_next = self.UAVs[i].p_next - 1
                    else:
                        self.UAVs[i].p = 0
                        self.UAVs[i].p_next = 1
                        self.UAVs[i].position = [self.UAVs[i].trajectory[self.UAVs[i].p_next] // 25, self.UAVs[i].trajectory[self.UAVs[i].p_next] % 25]
                        self.UAVs[i].p = self.UAVs[i].p + 1
                        self.UAVs[i].p_next = self.UAVs[i].p_next + 1

    def test_channel(self):
        self.n_step = 0
        self.UAVs = []
        n_UAV = 32
        self.n_UAV = n_UAV
        self.add_new_UAVs_by_number(int(self.n_UAV))
        step = 1000
        for i in range(step):
            actions_power = np.random.randint(0, 16, (n_UAV, 3, 2))
            for k in range(self.n_UAV):
                for j in range(3):
                    self.renew_positions()
            positions = [c.position for c in self.UAVs]
            self.update_large_fading(positions)
            self.update_small_fading()
            print("Time step: ", i)
            print(" ============== V2I ===========")
            print("Path Loss: ", self.U2Ichannels.PathLoss)
            print("Shadow:",  self.U2Ichannels.Shadow)
            print("Fast Fading: ",  self.U2Ichannels.FastFading)
            print(" ============== V2V ===========")
            print("Path Loss: ", self.U2Uchannels.PathLoss[0:3])
            print("Shadow:", self.U2Uchannels.Shadow[0:3])
            print("Fast Fading: ", self.U2Uchannels.FastFading[0:3])

    def update_large_fading(self, positions):
        self.U2Ichannels.update_positions(positions)
        self.U2Uchannels.update_positions(positions)
        self.U2Ichannels.update_pathloss()
        self.U2Uchannels.update_pathloss()
        delta_distance = 0.005 * np.asarray(c.velocity for c in self.UAVs)
        self.U2Ichannels.update_shadow(delta_distance)
        self.U2Uchannels.update_shadow(delta_distance)

    def update_small_fading(self):
        self.U2Ichannels.update_fast_fading()
        self.U2Uchannels.update_fast_fading()

    def renew_neighbor(self):
        # ==========================================
        # update the neighbors of each UAV.
        # ==========================================
        for i in range(len(self.UAVs)):
            self.UAVs[i].neighbors = []
            self.UAVs[i].actions = []
        Distance = np.zeros((len(self.UAVs), len(self.UAVs)))
        z = np.array([[complex(c.position[0], c.position[1]) for c in self.UAVs]])
        Distance = abs(z.T-z)
        for i in range(len(self.UAVs)):
            sort_idx = np.argsort(Distance[:, i])
            for j in range(3):
                self.UAVs[i].neighbors.append(sort_idx[j+1])
            destination = np.random.choice(sort_idx[1:int(len(sort_idx)/5)], 3, replace=False)
            self.UAVs[i].destinations = destination

    def renew_channel(self):
        # ===========================================================================
        # This function updates all the channels including U2U and U2I channels
        # =============================================================================
        positions = np.asarray([c.position for c in self.UAVs]) * 100
        self.U2Ichannels.update_positions(positions)
        self.U2Uchannels.update_positions(positions)
        self.U2Ichannels.update_pathloss()
        self.U2Uchannels.update_pathloss()
        delta_distance = 0.005 * np.asarray([c.velocity for c in self.UAVs])
        self.U2Ichannels.update_shadow(delta_distance)
        self.U2Uchannels.update_shadow(delta_distance)
        self.U2U_channels_abs = self.U2Uchannels.PathLoss + self.U2Uchannels.Shadow + 50 * np.identity(len(self.UAVs))
        self.U2I_channels_abs = self.U2Ichannels.PathLoss + self.U2Ichannels.Shadow_Los

    def renew_channels_fastfading(self):
        # =======================================================================
        # This function updates all the channels including U2U and U2I channels
        # =========================================================================
        self.renew_channel()
        self.U2Ichannels.update_fast_fading()
        self.U2Uchannels.update_fast_fading()
        U2U_channels_with_fastfading = np.repeat(self.U2U_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.U2U_channels_with_fastfading = U2U_channels_with_fastfading + self.U2Uchannels.FastFading
        U2I_channels_with_fastfading = np.repeat(self.U2I_channels_abs[:, np.newaxis], self.n_RB, axis=1)
        self.U2I_channels_with_fastfading = U2I_channels_with_fastfading + self.U2Ichannels.FastFading

    def Compute_Performance_Reward_fast_fading_with_power(self, actions_power):
        actions = actions_power.copy()[:, :, 0]          # the channel_selection_part
        power_selection = actions_power.copy()[:, :, 1]  # the power_selection_part
        Rate = np.zeros(len(self.UAVs))
        Interference = np.zeros(self.n_RB)               # U2U signal interference to U2I links
        for i in range(len(self.UAVs)):
            for j in range(len(actions[i, :])):
                if not self.activate_links[i, j]:
                    continue
                Interference[actions[i][j]] += 10**((self.U2U_power_dB_List[power_selection[i, j]] - self.U2I_channels_with_fastfading[i, actions[i, j]] + self.uavAntGain + self.bsAntGain - self.bsNoiseFigure)/10)  # fast fading

        self.U2I_Interference = Interference + self.sig2
        U2U_Interference = np.zeros((len(self.UAVs), 3))
        U2U_Signal = np.zeros((len(self.UAVs), 3))
        
        # remove the effects of none active links
        actions[(np.logical_not(self.activate_links))] = -1
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            for j in range(len(indexes)):
                receiver_j = self.UAVs[indexes[j, 0]].destinations[indexes[j, 1]]
                U2U_Signal[indexes[j, 0], indexes[j, 1]] = 10**((self.U2U_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]] - self.U2U_channels_with_fastfading[indexes[j][0]][receiver_j][i] + 2*self.uavAntGain - self.uavNoiseFigure)/10)
                if i < self.n_UAV:
                    U2U_Interference[indexes[j, 0], indexes[j, 1]] += 10**((self.U2I_power_dB - self.U2U_channels_with_fastfading[i][receiver_j][i] + 2*self.uavAntGain - self.uavNoiseFigure)/10)  # U2I links interference to U2U links
                for k in range(j+1, len(indexes)):                  # computer the peer U2U links
                    receiver_k = self.UAVs[indexes[k][0]].destinations[indexes[k][1]]
                    U2U_Interference[indexes[j, 0], indexes[j, 1]] += 10**((self.U2U_power_dB_List[power_selection[indexes[k, 0], indexes[k, 1]]] - self.U2U_channels_with_fastfading[indexes[k][0]][receiver_j][i] + 2*self.uavAntGain - self.uavNoiseFigure)/10)
                    U2U_Interference[indexes[k, 0], indexes[k, 1]] += 10**((self.U2U_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]] - self.U2U_channels_with_fastfading[indexes[j][0]][receiver_k][i] + 2*self.uavAntGain - self.uavNoiseFigure)/10)

        self.U2U_Interference = U2U_Interference + self.sig2
        U2U_Rate = np.zeros(self.activate_links.shape)
        U2U_Rate[self.activate_links] = np.log2(1 + np.divide(U2U_Signal[self.activate_links], self.U2U_Interference[self.activate_links]))

        U2I_Signals = self.U2I_power_dB - self.U2I_channels_abs[0:min(self.n_RB, self.n_UAV)] + self.uavAntGain + self.bsAntGain - self.bsNoiseFigure
        U2I_Rate = np.log2(1 + np.divide(10**(U2I_Signals/10), self.U2I_Interference[0:min(self.n_RB, self.n_UAV)]))

        # -- compute the latency constraints --
        self.demand -= U2U_Rate * self.update_time_test * 1500   # decrease the demand
        self.test_time_count -= self.update_time_test            # compute the time left for estimation
        self.individual_time_limit -= self.update_time_test      # compute the time left for individual U2U transmission
        self.individual_time_interval -= self.update_time_test   # compute the time interval left for next transmission

        # --- update the demand ---
        
        new_active = self.individual_time_interval <= 0
        self.activate_links[new_active] = True
        self.individual_time_interval[new_active] = np.random.exponential(0.02, self.individual_time_interval[new_active].shape) + self.U2U_limit
        self.individual_time_limit[new_active] = self.U2U_limit
        self.demand[new_active] = self.demand_amount

        # -- update the statistics---
        early_finish = np.multiply(self.demand <= 0, self.activate_links)        
        unqulified = np.multiply(self.individual_time_limit <= 0, self.activate_links)
        self.activate_links[np.add(early_finish, unqulified)] = False 
        # print('number of activate links is', np.sum(self.activate_links))
        self.success_transmission += np.sum(early_finish)
        self.failed_transmission += np.sum(unqulified)
        success_percentage = self.success_transmission/(self.failed_transmission + self.success_transmission + 0.0001)
        return U2I_Rate, U2U_Rate, success_percentage

    def Compute_Performance_Reward_fast_fading_with_power_asyn(self, actions_power):
        actions = actions_power[:, :, 0]           # the channel_selection_part
        power_selection = actions_power[:, :, 1]   # the power_selection_part
        Interference = np.zeros(self.n_RB)         # Calculate the interference from U2U to U2I
        for i in range(len(self.UAVs)):
            for j in range(len(actions[i, :])):
                if not self.activate_links[i, j]:
                    continue
                Interference[actions[i][j]] += 10**((self.U2U_power_dB_List[power_selection[i, j]] - \
                                                     self.U2I_channels_with_fastfading[i, actions[i, j]] + \
                                                     self.uavAntGain + self.bsAntGain - self.bsNoiseFigure)/10)
        self.U2I_Interference = Interference + self.sig2
        U2U_Interference = np.zeros((len(self.UAVs), 3))
        U2U_Signal = np.zeros((len(self.UAVs), 3))
        Interfence_times = np.zeros((len(self.UAVs), 3))
        actions[(np.logical_not(self.activate_links))] = -1
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            for j in range(len(indexes)):
                receiver_j = self.UAVs[indexes[j, 0]].destinations[indexes[j, 1]]
                U2U_Signal[indexes[j, 0], indexes[j, 1]] = 10**((self.U2U_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]] -\
                self.U2U_channels_with_fastfading[indexes[j][0]][receiver_j][i] + 2*self.uavAntGain - self.uavNoiseFigure)/10)
                if i < self.n_UAV:
                    U2U_Interference[indexes[j, 0], indexes[j, 1]] += 10**((self.U2I_power_dB - \
                    self.U2U_channels_with_fastfading[i][receiver_j][i] + 2*self.uavAntGain - self.uavNoiseFigure)/10)  # U2I links interference to U2U links
                for k in range(j+1, len(indexes)):
                    receiver_k = self.UAVs[indexes[k][0]].destinations[indexes[k][1]]
                    U2U_Interference[indexes[j, 0], indexes[j, 1]] += 10**((self.U2U_power_dB_List[power_selection[indexes[k, 0], indexes[k, 1]]] -\
                    self.U2U_channels_with_fastfading[indexes[k][0]][receiver_j][i] + 2*self.uavAntGain - self.uavNoiseFigure)/10)
                    U2U_Interference[indexes[k, 0], indexes[k, 1]] += 10**((self.U2U_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]] - \
                    self.U2U_channels_with_fastfading[indexes[j][0]][receiver_k][i] + 2*self.uavAntGain - self.uavNoiseFigure)/10)
                    Interfence_times[indexes[j, 0], indexes[j, 1]] += 1
                    Interfence_times[indexes[k, 0], indexes[k, 1]] += 1

        self.U2U_Interference = U2U_Interference + self.sig2
        U2U_Rate = np.log2(1 + np.divide(U2U_Signal, self.U2U_Interference))
        U2I_Signals = self.U2I_power_dB-self.U2I_channels_abs[0:min(self.n_RB, self.n_UAV)] + self.uavAntGain + self.bsAntGain - self.bsNoiseFigure
        U2I_Rate = np.log2(1 + np.divide(10**(U2I_Signals/10), self.U2I_Interference[0:min(self.n_RB, self.n_UAV)]))
        
        # -- compute the latency constraits --
        self.demand -= U2U_Rate * self.update_time_asyn * 1500  # decrease the demand
        self.test_time_count -= self.update_time_asyn          # compute the time left for estimation
        self.individual_time_limit -= self.update_time_asyn    # compute the time left for individual U2U transmission
        self.individual_time_interval -= self.update_time_asyn  # compute the time interval left for next transmission

        # --- update the demand ---
        new_active = self.individual_time_interval <= 0
        self.activate_links[new_active] = True
        self.individual_time_interval[new_active] = np.random.exponential(0.02, self.individual_time_interval[new_active].shape) + self.U2U_limit
        self.individual_time_limit[new_active] = self.U2U_limit
        self.demand[new_active] = self.demand_amount
        
        # -- update the statistics---
        early_finish = np.multiply(self.demand <= 0, self.activate_links)
        unqulified = np.multiply(self.individual_time_limit <= 0, self.activate_links)
        self.activate_links[np.add(early_finish, unqulified)] = False
        self.success_transmission += np.sum(early_finish)
        self.failed_transmission += np.sum(unqulified)
        success_percent = self.success_transmission / (self.failed_transmission + self.success_transmission + 0.0001)
        return U2I_Rate, U2U_Rate, success_percent

    def Compute_PSO_rate(self, actions_power):
        actions = actions_power[:, :, 0]          # the channel_selection_part
        power_selection = actions_power[:, :, 1]  # the power_selection_part
        Interference = np.zeros(self.n_RB)        # Calculate the interference from U2U to U2I
        for i in range(len(self.UAVs)):
            for j in range(len(actions[i, :])):
                if not self.activate_links_PSO[i, j]:
                    continue
                Interference[actions[i][j]] += 10 ** ((self.U2U_power_dB_List[power_selection[i, j]] - \
                                                       self.U2I_channels_with_fastfading[i, actions[i, j]] + \
                                                       self.uavAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.U2I_Interference = Interference + self.sig2
        U2U_Interference = np.zeros((len(self.UAVs), 3))
        U2U_Signal = np.zeros((len(self.UAVs), 3))
        Interfence_times = np.zeros((len(self.UAVs), 3))
        actions[(np.logical_not(self.activate_links_PSO))] = -1
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            for j in range(len(indexes)):
                receiver_j = self.UAVs[indexes[j, 0]].destinations[indexes[j, 1]]
                U2U_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** (
                            (self.U2U_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]] - \
                             self.U2U_channels_with_fastfading[indexes[j][0]][receiver_j][
                                 i] + 2 * self.uavAntGain - self.uavNoiseFigure) / 10)
                if i < self.n_UAV:
                    U2U_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.U2I_power_dB - \
                                                                              self.U2U_channels_with_fastfading[i][
                                                                                  receiver_j][
                                                                                  i] + 2 * self.uavAntGain - self.uavNoiseFigure) / 10)  # U2I links interference to U2U links
                for k in range(j + 1, len(indexes)):
                    receiver_k = self.UAVs[indexes[k][0]].destinations[indexes[k][1]]
                    U2U_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** (
                                (self.U2U_power_dB_List[power_selection[indexes[k, 0], indexes[k, 1]]] - \
                                 self.U2U_channels_with_fastfading[indexes[k][0]][receiver_j][
                                     i] + 2 * self.uavAntGain - self.uavNoiseFigure) / 10)
                    U2U_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** (
                                (self.U2U_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]] - \
                                 self.U2U_channels_with_fastfading[indexes[j][0]][receiver_k][
                                     i] + 2 * self.uavAntGain - self.uavNoiseFigure) / 10)
                    Interfence_times[indexes[j, 0], indexes[j, 1]] += 1
                    Interfence_times[indexes[k, 0], indexes[k, 1]] += 1

        self.U2U_Interference = U2U_Interference + self.sig2
        U2U_Rate = np.log2(1 + np.divide(U2U_Signal, self.U2U_Interference))
        U2I_Signals = self.U2I_power_dB - self.U2I_channels_abs[0:min(self.n_RB,
                                                                      self.n_UAV)] + self.uavAntGain + self.bsAntGain - self.bsNoiseFigure
        U2I_Rate = np.log2(1 + np.divide(10 ** (U2I_Signals / 10), self.U2I_Interference[0:min(self.n_RB, self.n_UAV)]))

        # -- compute the latency constraits --
        self.demand_PSO -= U2U_Rate * self.update_time_asyn_PSO * 1500  # decrease the demand
        self.test_time_count_PSO -= self.update_time_asyn_PSO           # compute the time left for estimation
        self.individual_time_limit_PSO -= self.update_time_asyn_PSO     # compute the time left for individual U2U transmission
        self.individual_time_interval_PSO -= self.update_time_asyn_PSO  # compute the time interval left for next transmission

        # --- update the demand ---
        new_active_PSO = self.individual_time_interval_PSO <= 0
        self.activate_links_PSO[new_active_PSO] = True
        self.individual_time_interval_PSO[new_active_PSO] = np.random.exponential(0.02, self.individual_time_interval_PSO[new_active_PSO].shape) + self.U2U_limit_PSO
        self.individual_time_limit_PSO[new_active_PSO] = self.U2U_limit_PSO
        self.demand_PSO[new_active_PSO] = self.demand_amount_PSO

        # -- update the statistics---
        early_finish_PSO = np.multiply(self.demand_PSO <= 0, self.activate_links_PSO)
        unqulified_PSO = np.multiply(self.individual_time_limit_PSO <= 0, self.activate_links_PSO)
        self.activate_links_PSO[np.add(early_finish_PSO, unqulified_PSO)] = False
        self.success_transmission_PSO += np.sum(early_finish_PSO)
        self.failed_transmission_PSO += np.sum(unqulified_PSO)
        success_percent_PSO = self.success_transmission_PSO / (self.failed_transmission_PSO + self.success_transmission_PSO + 0.0001)
        return U2I_Rate, U2U_Rate, success_percent_PSO

    def Compute_Performance_Reward_Batch(self, actions_power, idx):
        # ==================================================
        # ------------- Used for Training ----------------
        # ==================================================
        actions = actions_power.copy()[:, :, 0]                     # the channel_selection_part
        power_selection = actions_power.copy()[:, :, 1]             # the power_selection_part
        U2U_Interference = np.zeros((len(self.UAVs), 3))
        U2U_Signal = np.zeros((len(self.UAVs), 3))
        Interfence_times = np.zeros((len(self.UAVs), 3))    # 3 neighbors
        origin_channel_selection = actions[idx[0], idx[1]]
        actions[idx[0], idx[1]] = 100  # something not ralevant
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            # print('index',indexes)
            for j in range(len(indexes)):
                receiver_j = self.UAVs[indexes[j, 0]].destinations[indexes[j, 1]]
                U2U_Signal[indexes[j, 0], indexes[j, 1]] = 10**((self.U2U_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]] - self.U2U_channels_with_fastfading[indexes[j, 0], receiver_j, i] + 2*self.uavAntGain - self.uavNoiseFigure)/10)
                U2U_Interference[indexes[j, 0], indexes[j, 1]] += 10**((self.U2I_power_dB - self.U2U_channels_with_fastfading[i, receiver_j, i] + \
                2*self.uavAntGain - self.uavNoiseFigure)/10)
                
                for k in range(j+1, len(indexes)):
                    receiver_k = self.UAVs[indexes[k, 0]].destinations[indexes[k, 1]]
                    U2U_Interference[indexes[j, 0], indexes[j, 1]] += 10**((self.U2U_power_dB_List[power_selection[indexes[k, 0], indexes[k, 1]]] - \
                    self.U2U_channels_with_fastfading[indexes[k, 0], receiver_j, i] + 2*self.uavAntGain - self.uavNoiseFigure)/10)
                    U2U_Interference[indexes[k, 0], indexes[k, 1]] += 10**((self.U2U_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]] - \
                    self.U2U_channels_with_fastfading[indexes[j, 0], receiver_k, i] + 2*self.uavAntGain - self.uavNoiseFigure)/10)
                    Interfence_times[indexes[j, 0], indexes[j, 1]] += 1
                    Interfence_times[indexes[k, 0], indexes[k, 1]] += 1
                    
        self.U2U_Interference = U2U_Interference + self.sig2
        U2U_Rate_list = np.zeros((self.n_RB, len(self.U2U_power_dB_List)))
        Deficit_list = np.zeros((self.n_RB, len(self.U2U_power_dB_List)))
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)

            U2U_Signal_temp = U2U_Signal.copy()
            receiver_k = self.UAVs[idx[0]].destinations[idx[1]]
            for power_idx in range(len(self.U2U_power_dB_List)):
                U2U_Interference_temp = U2U_Interference.copy()
                U2U_Signal_temp[idx[0], idx[1]] = 10**((self.U2U_power_dB_List[power_idx] - \
                self.U2U_channels_with_fastfading[idx[0], self.UAVs[idx[0]].destinations[idx[1]], i] + 2*self.uavAntGain - self.uavNoiseFigure)/10)
                U2U_Interference_temp[idx[0], idx[1]] += 10**((self.U2I_power_dB - \
                self.U2U_channels_with_fastfading[i, self.UAVs[idx[0]].destinations[idx[1]], i] + 2*self.uavAntGain - self.uavNoiseFigure)/10)
                for j in range(len(indexes)):
                    receiver_j = self.UAVs[indexes[j, 0]].destinations[indexes[j, 1]]
                    U2U_Interference_temp[idx[0], idx[1]] += 10**((self.U2U_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]] -\
                    self.U2U_channels_with_fastfading[indexes[j, 0], receiver_k, i] + 2*self.uavAntGain - self.uavNoiseFigure)/10)
                    U2U_Interference_temp[indexes[j, 0], indexes[j, 1]] += 10**((self.U2U_power_dB_List[power_idx]-\
                    self.U2U_channels_with_fastfading[idx[0], receiver_j, i] + 2*self.uavAntGain - self.uavNoiseFigure)/10)
                U2U_Rate_cur = np.log2(1 + np.divide(U2U_Signal_temp, U2U_Interference_temp))
                if (origin_channel_selection == i) and (power_selection[idx[0], idx[1]] == power_idx):
                    U2U_Rate = U2U_Rate_cur.copy()

                U2U_Rate_list[i, power_idx] = np.sum(U2U_Rate_cur)
                Deficit_list[i, power_idx] = 0 - 1 * np.sum(np.maximum(np.zeros(U2U_Signal_temp.shape), (self.demand - self.individual_time_limit * U2U_Rate_cur * 1500)))
        Interference = np.zeros(self.n_RB)
        U2I_Rate_list = np.zeros((self.n_RB, len(self.U2U_power_dB_List)))
        for i in range(len(self.UAVs)):
            for j in range(len(actions[i, :])):
                if (i == idx[0] and j == idx[1]):
                    continue
                Interference[actions[i][j]] += 10**((self.U2U_power_dB_List[power_selection[i, j]] -
                                                     self.U2I_channels_with_fastfading[i, actions[i][j]] + self.uavAntGain + self.bsAntGain - self.bsNoiseFigure)/10)
        U2I_Interference = Interference + self.sig2
        for i in range(self.n_RB):            
            for j in range(len(self.U2U_power_dB_List)):
                U2I_Interference_temp = U2I_Interference.copy()
                U2I_Interference_temp[i] += 10**((self.U2U_power_dB_List[j] - self.U2I_channels_with_fastfading[idx[0], i] + self.uavAntGain + self.bsAntGain - self.bsNoiseFigure)/10)
                U2I_Rate_list[i, j] = np.sum(np.log2(1 + np.divide(10**((self.U2I_power_dB + self.uavAntGain + self.bsAntGain -
                                                                         self.bsNoiseFigure-self.U2I_channels_abs[0:min(self.n_RB, self.n_UAV)])/10), U2I_Interference_temp[0:min(self.n_RB, self.n_UAV)])))
                     
        self.demand -= U2U_Rate * self.update_time_train * 1500
        self.test_time_count -= self.update_time_train
        self.individual_time_limit -= self.update_time_train
        self.individual_time_limit[np.add(self.individual_time_limit <= 0,  self.demand < 0)] = self.U2U_limit
        self.demand[self.demand < 0] = self.demand_amount
        if self.test_time_count == 0:
            self.test_time_count = 10
        return U2I_Rate_list, Deficit_list, self.individual_time_limit[idx[0], idx[1]]

    def Compute_PSO_fitness(self, actions_power, idx):
        self.demand_amount = self.demand_PSO
        self.demand = self.demand_amount * np.ones((self.n_UAV, 3))
        self.test_time_count = self.test_time_count_PSO
        self.U2U_limit = self.U2U_limit_PSO  # 100 ms U2U toleratable latency
        self.individual_time_limit = self.U2U_limit * np.ones((self.n_UAV, 3))
        actions = actions_power.copy()[:, :, 0].astype(int)          # the channel_selection_part
        power_selection = actions_power.copy()[:, :, 1].astype(int)  # the power_selection_part
        U2U_Interference = np.zeros((len(self.UAVs), 3))
        U2U_Signal = np.zeros((len(self.UAVs), 3))
        Interfence_times = np.zeros((len(self.UAVs), 3))  # 3 neighbors
        origin_channel_selection = actions[idx[0], idx[1]]
        actions[idx[0], idx[1]] = 100  # something not ralevant
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            # print('index',indexes)
            for j in range(len(indexes)):
                receiver_j = self.UAVs[indexes[j, 0]].destinations[indexes[j, 1]]
                U2U_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.U2U_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]] - self.U2U_channels_with_fastfading[indexes[j, 0], receiver_j, i] + 2 * self.uavAntGain - self.uavNoiseFigure) / 10)
                U2U_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** (
                            (self.U2I_power_dB - self.U2U_channels_with_fastfading[i, receiver_j, i] + \
                             2 * self.uavAntGain - self.uavNoiseFigure) / 10)  # interference from the U2I links

                for k in range(j + 1, len(indexes)):
                    receiver_k = self.UAVs[indexes[k, 0]].destinations[indexes[k, 1]]
                    U2U_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** (
                                (self.U2U_power_dB_List[power_selection[indexes[k, 0], indexes[k, 1]]] - \
                                 self.U2U_channels_with_fastfading[
                                     indexes[k, 0], receiver_j, i] + 2 * self.uavAntGain - self.uavNoiseFigure) / 10)
                    U2U_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** (
                                (self.U2U_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]] - \
                                 self.U2U_channels_with_fastfading[
                                     indexes[j, 0], receiver_k, i] + 2 * self.uavAntGain - self.uavNoiseFigure) / 10)
                    Interfence_times[indexes[j, 0], indexes[j, 1]] += 1
                    Interfence_times[indexes[k, 0], indexes[k, 1]] += 1

        self.U2U_Interference = U2U_Interference + self.sig2
        U2U_Rate_list = np.zeros((self.n_RB, len(self.U2U_power_dB_List)))
        Deficit_list = np.zeros((self.n_RB, len(self.U2U_power_dB_List)))
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            U2U_Signal_temp = U2U_Signal.copy()
            receiver_k = self.UAVs[idx[0]].destinations[idx[1]]
            for power_idx in range(len(self.U2U_power_dB_List)):
                U2U_Interference_temp = U2U_Interference.copy()
                U2U_Signal_temp[idx[0], idx[1]] = 10 ** ((self.U2U_power_dB_List[power_idx] - \
                                                          self.U2U_channels_with_fastfading[
                                                              idx[0], self.UAVs[idx[0]].destinations[idx[
                                                                  1]], i] + 2 * self.uavAntGain - self.uavNoiseFigure) / 10)
                U2U_Interference_temp[idx[0], idx[1]] += 10 ** ((self.U2I_power_dB - \
                                                                 self.U2U_channels_with_fastfading[
                                                                     i, self.UAVs[idx[0]].destinations[idx[
                                                                         1]], i] + 2 * self.uavAntGain - self.uavNoiseFigure) / 10)
                for j in range(len(indexes)):
                    receiver_j = self.UAVs[indexes[j, 0]].destinations[indexes[j, 1]]
                    U2U_Interference_temp[idx[0], idx[1]] += 10 ** (
                                (self.U2U_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]] - \
                                 self.U2U_channels_with_fastfading[
                                     indexes[j, 0], receiver_k, i] + 2 * self.uavAntGain - self.uavNoiseFigure) / 10)
                    U2U_Interference_temp[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.U2U_power_dB_List[power_idx] - \
                                                                                   self.U2U_channels_with_fastfading[
                                                                                       idx[
                                                                                           0], receiver_j, i] + 2 * self.uavAntGain - self.uavNoiseFigure) / 10)
                U2U_Rate_cur = np.log2(1 + np.divide(U2U_Signal_temp, U2U_Interference_temp))
                if (origin_channel_selection == i) and (power_selection[idx[0], idx[1]] == power_idx):
                    U2U_Rate = U2U_Rate_cur.copy()

                U2U_Rate_list[i, power_idx] = np.sum(U2U_Rate_cur)
                Deficit_list[i, power_idx] = 0 - 1 * np.sum(np.maximum(np.zeros(U2U_Signal_temp.shape), (
                            self.demand - self.individual_time_limit * U2U_Rate_cur * 1500)))
        Interference = np.zeros(self.n_RB)
        U2I_Rate_list = np.zeros((self.n_RB, len(self.U2U_power_dB_List)))
        for i in range(len(self.UAVs)):
            for j in range(len(actions[i, :])):
                if (i == idx[0] and j == idx[1]):
                    continue
                Interference[actions[i][j]] += 10 ** ((self.U2U_power_dB_List[power_selection[i, j]] -
                                                       self.U2I_channels_with_fastfading[i, actions[i][
                                                           j]] + self.uavAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        U2I_Interference = Interference + self.sig2
        for i in range(self.n_RB):
            for j in range(len(self.U2U_power_dB_List)):
                U2I_Interference_temp = U2I_Interference.copy()
                U2I_Interference_temp[i] += 10 ** ((self.U2U_power_dB_List[j] - self.U2I_channels_with_fastfading[
                    idx[0], i] + self.uavAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
                U2I_Rate_list[i, j] = np.sum(
                    np.log2(1 + np.divide(10 ** ((self.U2I_power_dB + self.uavAntGain + self.bsAntGain -
                                                  self.bsNoiseFigure - self.U2I_channels_abs[
                                                                       0:min(self.n_RB, self.n_UAV)]) / 10),
                                          U2I_Interference_temp[0:min(self.n_RB, self.n_UAV)])))

        self.demand -= U2U_Rate * self.update_time_train * 1500
        self.test_time_count -= self.update_time_train
        self.individual_time_limit -= self.update_time_train

        return U2I_Rate_list, Deficit_list, self.individual_time_limit[idx[0], idx[1]]

    def Compute_Interference(self, actions):
        # ====================================================
        # Compute the Interference to each channel_selection
        # ====================================================
        U2U_Interference = np.zeros((len(self.UAVs), 3, self.n_RB)) + self.sig2
        if len(actions.shape) == 3:
            channel_selection = actions.copy()[:, :, 0]
            power_selection = actions.copy()[:, :, 1]
            channel_selection[np.logical_not(self.activate_links)] = -1
            for i in range(self.n_RB):
                for k in range(len(self.UAVs)):
                    for m in range(len(channel_selection[k, :])):
                        U2U_Interference[k, m, i] += 10 ** ((self.U2I_power_dB - self.U2U_channels_with_fastfading[i][self.UAVs[k].destinations[m]][i] + \
                        2 * self.uavAntGain - self.uavNoiseFigure)/10)
            for i in range(len(self.UAVs)):
                for j in range(len(channel_selection[i, :])):
                    for k in range(len(self.UAVs)):
                        for m in range(len(channel_selection[k, :])):
                            if i == k or channel_selection[i, j] >= 0:
                                continue
                            U2U_Interference[k, m, channel_selection[i, j]] += 10**((self.U2U_power_dB_List[power_selection[i,j]] -\
                            self.U2U_channels_with_fastfading[i][self.UAVs[k].destinations[m]][channel_selection[i, j]] + 2*self.uavAntGain - self.uavNoiseFigure)/10)

        self.U2U_Interference_all = 10 * np.log10(U2U_Interference)

    def renew_demand(self):
        # generate a new demand of a U2U
        self.demand = self.demand_amount*np.ones((self.n_RB, 3))
        self.time_limit = 10

    def act_for_training(self, actions, idx):
        # =============================================
        # This function gives rewards for training
        # ===========================================
        rewards_list = np.zeros(self.n_RB)
        action_temp = actions.copy()
        self.activate_links = np.ones((self.n_UAV, 3), dtype='bool')
        U2I_rewardlist, U2U_rewardlist, time_left = self.Compute_Performance_Reward_Batch(action_temp, idx)
        rewards_list = rewards_list.T.reshape([-1])
        U2I_rewardlist = U2I_rewardlist.T.reshape([-1])
        U2U_rewardlist = U2U_rewardlist.T.reshape([-1])
        U2I_reward = (U2I_rewardlist[actions[idx[0], idx[1], 0] + 16 * actions[idx[0], idx[1], 1]] -
                      np.min(U2I_rewardlist)) / (np.max(U2I_rewardlist) - np.min(U2I_rewardlist) + 0.000001)
        U2U_reward = (U2U_rewardlist[actions[idx[0], idx[1], 0] + 16 * actions[idx[0], idx[1], 1]] -
                      np.min(U2U_rewardlist)) / (np.max(U2U_rewardlist) - np.min(U2U_rewardlist) + 0.000001)
        lambdda = 0.1

        t = lambdda * U2I_reward + (1 - lambdda) * U2U_reward
        # print("time left", time_left)
        # return t
        if idx[1] == 2:
            self.renew_positions()
            self.renew_neighbor()
        self.renew_channels_fastfading()

        self.Compute_Interference(actions)

        return t - (self.U2U_limit-time_left)/self.U2U_limit

    def act_asyn(self, actions):
        self.n_step += 1
        reward = self.Compute_Performance_Reward_fast_fading_with_power_asyn(actions)
        if self.n_step % 10 == 0:
            self.renew_positions()
            self.renew_neighbor()
        self.renew_channels_fastfading()
        self.Compute_Interference(actions)
        return reward

    def act_play(self, actions):
        self.n_step += 1
        reward = self.Compute_Performance_Reward_fast_fading_with_power_asyn(actions)
        if self.n_step % 10 == 0:
            self.renew_positions()
            self.renew_neighbor()
        self.renew_channels_fastfading()
        self.Compute_Interference(actions)
        return reward

    def act_random(self, actions):
        # simulate the next state after the action is given
        self.n_step += 1
        reward_random = self.Compute_Performance_Reward_fast_fading_with_power(actions)
        if self.n_step % 10 == 0:
            self.renew_positions()
            self.renew_neighbor()
        self.renew_channels_fastfading()
        self.Compute_Interference(actions)
        return reward_random

    def act_pso(self, actions):
        # simulate the next state after the action is given
        self.n_step += 1
        reward_pso = self.Compute_PSO_rate(actions)
        self.renew_positions()
        self.renew_neighbor()
        self.renew_channels_fastfading()
        return reward_pso

    def new_random_game(self, n_UAV):
        # make a new game
        self.n_step = 0
        self.UAVs = []
        self.addresses = []
        if n_UAV > 0:
            self.n_UAV = n_UAV
        self.add_new_UAVs_by_number(int(self.n_UAV))
        self.U2Uchannels = U2Uchannels(self.n_UAV, self.n_RB)
        self.U2Ichannels = U2Ichannels(self.n_UAV, self.n_RB)
        self.renew_channels_fastfading()
        self.renew_neighbor()
        self.demand_amount = 30
        self.demand = self.demand_amount * np.ones((self.n_UAV, 3))
        self.test_time_count = 10
        self.U2U_limit = 0.1  # 100 ms U2U toleratable latency
        self.individual_time_limit = self.U2U_limit * np.ones((self.n_UAV, 3))
        self.individual_time_interval = np.random.exponential(0.05, (self.n_UAV, 3))
        self.UnsuccessfulLink = np.zeros((self.n_UAV, 3))
        self.success_transmission = 0
        self.failed_transmission = 0
        self.update_time_train = 0.01  # 10ms update time for the training
        self.update_time_test = 0.002  # 2ms update time for testing
        self.update_time_asyn = 0.0002  # 0.2 ms update one subset of the UAVs; for each UAV, the update time is 2 ms
        self.activate_links = np.zeros((self.n_UAV, 3), dtype='bool')

        self.demand_amount_PSO = 30
        self.demand_PSO = self.demand_amount_PSO * np.ones((self.n_UAV, 3))
        self.test_time_count_PSO = 10
        self.U2U_limit_PSO = 0.1  # 100 ms U2U toleratable latency
        self.individual_time_limit_PSO = self.U2U_limit_PSO * np.ones((self.n_UAV, 3))
        self.individual_time_interval_PSO = np.random.exponential(0.05, (self.n_UAV, 3))
        self.UnsuccessfulLink_PSO = np.zeros((self.n_UAV, 3))
        self.success_transmission_PSO = 0
        self.failed_transmission_PSO = 0
        self.update_time_train_PSO = 0.01  # 10ms update time for the training
        self.update_time_test_PSO = 0.002  # 2ms update time for testing
        self.update_time_asyn_PSO = 0.0002  # 0.2 ms update one subset of the UAVs; for each UAV, the update time is 2 ms
        self.activate_links_PSO = np.zeros((self.n_UAV, 3), dtype='bool')


if __name__ == "__main__":

    Env = Environ()
    Env.test_channel()

