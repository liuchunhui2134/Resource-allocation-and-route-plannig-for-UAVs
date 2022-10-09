from __future__ import print_function, division
import os
import time
import random
import numpy as np
from Environment import *
from base import BaseModel
from replay_memory import ReplayMemory
from utils import save_pkl, load_pkl
import tensorflow as tf
import matplotlib.pyplot as plt

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


class Agent(BaseModel):
    def __init__(self, config, environment, sess):
        self.sess = sess
        self.weight_dir = 'weight'        
        self.env = environment
        # self.history = History(self.config)
        model_dir = './Model/a.model'
        self.memory = ReplayMemory(model_dir) 
        self.max_step = 15000
        self.RB_number = 16
        self.num_UAV = len(self.env.UAVs)
        self.action_all_with_power = np.zeros([32, 3, 2], dtype='int32')
        self.action_all_with_power_training = np.zeros([32, 3, 2], dtype='int32')
        self.reward = []
        self.cost = []
        self.flag = 0
        self.learning_rate = 0.01
        self.learning_rate_minimum = 0.0001
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 500000
        self.target_q_update_step = 50
        self.discount = 0.6
        self.double_q = True
        self.build_dqn()          
        self.U2U_number = 3 * len(self.env.UAVs)    # every UAV need to communicate with 3 neighbors
        self.training = True

    def merge_action(self, idx, action):
        self.action_all_with_power[idx[0], idx[1], 0] = action % 16               # frequency
        self.action_all_with_power[idx[0], idx[1], 1] = int(np.floor(action/16))  # power

    def get_state(self, idx):
        # ===============
        # Get State from the environment
        # ===============
        UAV_number = len(self.env.UAVs)
        U2U_channel = (self.env.U2U_channels_with_fastfading[idx[0], self.env.UAVs[idx[0]].destinations[idx[1]], :]-80)/64
        U2I_channel = (self.env.U2I_channels_with_fastfading[idx[0], :] - 80)/64
        U2U_interference = (-self.env.U2U_Interference_all[idx[0], idx[1], :] - 80)/64
        NeiSelection = np.zeros(self.RB_number)
        for i in range(3):
            for j in range(3):
                if self.training:
                    NeiSelection[self.action_all_with_power_training[self.env.UAVs[idx[0]].neighbors[i], j, 0]] = 1
                else:
                    NeiSelection[self.action_all_with_power[self.env.UAVs[idx[0]].neighbors[i], j, 0]] = 1
                   
        for i in range(3):
            if i == idx[1]:
                continue
            if self.training:
                if self.action_all_with_power_training[idx[0], i, 0] >= 0:
                    NeiSelection[self.action_all_with_power_training[idx[0], i, 0]] = 1
            else:
                if self.action_all_with_power[idx[0], i, 0] >= 0:
                    NeiSelection[self.action_all_with_power[idx[0], i, 0]] = 1
        load_remaining = np.asarray([self.env.demand[idx[0], idx[1]]/self.env.demand_amount])
        time_remaining = np.asarray([self.env.individual_time_limit[idx[0], idx[1]]/self.env.U2U_limit])
        return np.concatenate((U2I_channel, U2U_interference, U2U_channel, NeiSelection, load_remaining, time_remaining))

    def predict(self, s_t,  step, test_ep=False):
        # ==========================
        #  Select actions
        # ======================
        ep = 1/(step/1000 + 1)
        if random.random() < ep and test_ep == False:   # epsion to balance the exporation and exploition
            action = np.random.randint(64)
        else:
            action = self.q_action.eval({self.s_t: [s_t]})[0]
            self.flag = 1
        return action

    def observe(self, prestate, state, reward, action):
        # -----------
        # Collect Data for Training
        # ---------
        self.memory.add(prestate, state, reward, action)  # add the state and the action and the reward to the memory
        # print(self.step)
        if self.step > 0:
            if self.step % 50 == 0:
                # print('Training')
                self.q_learning_mini_batch()            # training a mini batch
                # self.save_weight_to_pkl()
            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                # print("Update Target Q network:")
                self.update_target_q_network(1-1e-2)

    def train(self):
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_ep_reward = 0
        ep_reward, actions = [], []        
        mean_big = 0
        number_big = 0
        mean_not_big = 0
        number_not_big = 0
        self.env.new_random_game(32)
        a = 0
        for self.step in (range(0, 2001)):
            if self.step == 0:
                num_game, self.update_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_reward, actions = [], []               

            if self.step % 2000 == 1 and self.step > 1:
                self.env.new_random_game(32)
            print(self.step)
            state_old = self.get_state([0, 0])
            self.training = True
            for k in range(1):
                for i in range(len(self.env.UAVs)):
                    for j in range(3):
                        state_old = self.get_state([i, j])
                        action = self.predict(state_old, self.step, False)
                        self.action_all_with_power_training[i, j, 0] = action % 16
                        self.action_all_with_power_training[i, j, 1] = int(np.floor(action/16))
                        reward_train = self.env.act_for_training(self.action_all_with_power_training, [i, j])
                        self.reward.append(reward_train)
                        state_new = self.get_state([i, j])
                        self.observe(state_old, state_new, reward_train, action)

            if (self.step % 2000 == 0) and (self.step > 0):
                # testing 
                self.training = False
                number_of_game = 10
                if (self.step % 10000 == 0) and (self.step > 0):
                    number_of_game = 50 
                if self.step == 38000:
                    number_of_game = 100               
                U2I_Rate_list = np.zeros(number_of_game)
                U2U_Rate_list = np.zeros(number_of_game)
                success_percent_list = np.zeros(number_of_game)
                for game_idx in range(number_of_game):
                    self.env.new_random_game(self.num_UAV)
                    test_sample = 200
                    U2IRate_list = []
                    U2URate_list = []
                    print('test game idx:', game_idx)
                    for k in range(test_sample):
                        action_temp = self.action_all_with_power.copy()
                        for i in range(len(self.env.UAVs)):
                            self.action_all_with_power[i, :, 0] = -1
                            sorted_idx = np.argsort(self.env.individual_time_limit[i, :])
                            for j in sorted_idx:                   
                                state_old = self.get_state([i, j])
                                action = self.predict(state_old, self.step, True)
                                self.merge_action([i, j], action)

                            if i % (len(self.env.UAVs)/8) == 1:

                                action_temp = self.action_all_with_power.copy()
                                U2Ireward, U2Ureward, percent = self.env.act_asyn(action_temp)
                                U2IRate_list.append(np.sum(U2Ireward))
                                U2URate_list.append(np.sum(U2Ureward))

                            if i == 31:
                                action_temp = self.action_all_with_power.copy()
                                U2Ireward, U2Ureward, percent = self.env.act_asyn(action_temp)
                                U2IRate_list.append(np.sum(U2Ireward))
                                U2URate_list.append(np.sum(U2Ureward))

                    U2I_Rate_list[game_idx] = np.mean(np.asarray(U2IRate_list))
                    U2U_Rate_list[game_idx] = np.mean(np.asarray(U2URate_list))
                    success_percent_list[game_idx] = percent
                    # print("action is", self.action_all_with_power)
                    print('success probability is, ', percent)

                self.save_weight_to_pkl()
                print('The number of UAVs is ', len(self.env.UAVs))
                print('Mean of the U2I rate is that ', np.mean(U2I_Rate_list))
                print('Mean of the U2U rate is that ', np.mean(U2U_Rate_list))
                print('Mean of success percent is that ', np.mean(success_percent_list))

        plt.plot(np.arange(len(self.cost)), self.cost)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

        plt.plot(np.arange(len(self.reward)), self.reward)
        plt.ylabel('Reward')
        plt.xlabel('training steps')
        plt.show()
            
    def q_learning_mini_batch(self):

        # Training the DQN model

        s_t, s_t_plus_1, action, reward = self.memory.sample()
        t = time.time()        
        if self.double_q:       # double Q learning
            pred_action = self.q_action.eval({self.s_t: s_t_plus_1})       
            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({self.target_s_t: s_t_plus_1, self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]})            
            target_q_t = self.discount * q_t_plus_1_with_pred_action + reward
        else:
            q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})         
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
            target_q_t = self.discount * max_q_t_plus_1 + reward
        _, q_t, loss, w = self.sess.run([self.optim, self.q, self.loss, self.w], {self.target_q_t: target_q_t, self.action:action, self.s_t:s_t, self.learning_rate_step: self.step}) # training the network
        
        print('loss is ', loss)
        self.cost.append(loss)
        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1

    def build_dqn(self): 
        # --- Building the DQN -------
        self.w = {}
        self.t_w = {}        
        
        initializer = tf. truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        n_hidden_1 = 500
        n_hidden_2 = 250
        n_hidden_3 = 120
        n_input = 66
        n_output = 64

        def encoder(x):
            weights = {                    
                'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
                'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
                'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1)),
                'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1)),

                'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1)),
                'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1)),
                'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1)),
                'encoder_b4': tf.Variable(tf.truncated_normal([n_output], stddev=0.1)),
            
            }
            layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), weights['encoder_b1']))
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']), weights['encoder_b2']))
            layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']), weights['encoder_b3']))
            layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['encoder_h4']), weights['encoder_b4']))

            return layer_4, weights

        with tf.variable_scope('prediction'):
            self.s_t = tf.placeholder('float32', [None, n_input])
            self.q, self.w = encoder(self.s_t)
            self.q_action = tf.argmax(self.q, dimension=1)

        with tf.variable_scope('target'):
            self.target_s_t = tf.placeholder('float32', [None, n_input])
            self.target_q, self.target_w = encoder(self.target_s_t)
            self.target_q_idx = tf.placeholder('int32', [None, None], 'output_idx')
            self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}
            for name in self.w.keys():
                print('name in self w keys', name)
                self.t_w_input[name] = tf.placeholder('float32', self.target_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.target_w[name].assign(self.t_w_input[name])       
        
        def clipped_error(x):
            try:
                return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
            except:
                return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', None, name='target_q_t')
            self.action = tf.placeholder('int32', None, name='action')
            action_one_hot = tf.one_hot(self.action, n_output, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')
            self.delta = self.target_q_t - q_acted
            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.reduce_mean(tf.square(self.delta), name='loss')
            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum, tf.train.exponential_decay(self.learning_rate, self.learning_rate_step, self.learning_rate_decay_step, self.learning_rate_decay, staircase=True))
            self.optim = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss) 
        
        tf.initialize_all_variables().run()
        self.update_target_q_network(1-1e-2)

    def update_target_q_network(self, var):
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: var*self.target_w[name].eval() + (1-var)*self.w[name].eval()})
        
    def save_weight_to_pkl(self): 
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)
        for name in self.w.keys():
            save_pkl(self.w[name].eval(), os.path.join(self.weight_dir, "%s.pkl" % name))

    def load_weight_from_pkl(self):
        with tf.variable_scope('load_pred_from_pkl'):
            self.w_input = {}
            self.w_assign_op = {}
            for name in self.w.keys():
                self.w_input[name] = tf.placeholder('float32')
                self.w_assign_op[name] = self.w[name].assign(self.w_input[name])
        for name in self.w.keys():
            self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})
        self.update_target_q_network(1-1e-2)
      
    def play(self, n_step=100, n_episode=100, test_ep=None, render=False):
        number_of_game = 10
        U2I_Rate_list = np.zeros(number_of_game)
        U2U_Rate_list = np.zeros(number_of_game)
        success_percent_list = np.zeros(number_of_game)
        self.load_weight_from_pkl()
        self.training = False

        for game_idx in range(number_of_game):
            self.env.new_random_game(self.num_UAV)
            test_sample = 200
            U2IRate_list = []
            U2URate_list = []
            print('test game idx:', game_idx)
            print('The number of UAVs is ', len(self.env.UAVs))
            time_left_list = []
            power_select_list_0 = []
            power_select_list_1 = []
            power_select_list_2 = []
            power_select_list_3 = []

            for k in range(test_sample):
                action_temp = self.action_all_with_power.copy()
                for i in range(len(self.env.UAVs)):
                    self.action_all_with_power[i, :, 0] = -1
                    sorted_idx = np.argsort(self.env.individual_time_limit[i, :])
                    for j in sorted_idx:
                        state_old = self.get_state([i, j])
                        time_left_list.append(state_old[-1])
                        action = self.predict(state_old, 0, True)
                        if state_old[-1] <= 0:
                            continue
                        power_selection = int(np.floor(action/16))
                        if power_selection == 0:
                            power_select_list_0.append(state_old[-1])

                        if power_selection == 1:
                            power_select_list_1.append(state_old[-1])
                        if power_selection == 2:
                            power_select_list_2.append(state_old[-1])
                        if power_selection == 3:
                            power_select_list_3.append(state_old[-1])

                        self.merge_action([i, j], action)

                    if i % (len(self.env.UAVs)/8) == 1:

                        action_temp = self.action_all_with_power.copy()
                        U2Ireward, U2Ureward, percent = self.env.act_asyn(action_temp)
                        U2IRate_list.append(np.sum(U2Ireward))
                        U2URate_list.append(np.sum(U2Ureward))
                    if i == 31:
                        action_temp = self.action_all_with_power.copy()
                        U2Ireward, U2Ureward, percent = self.env.act_asyn(action_temp)
                        U2IRate_list.append(np.sum(U2Ireward))
                        U2URate_list.append(np.sum(U2Ureward))

            number_0, bin_edges = np.histogram(power_select_list_0, bins=10)

            number_1, bin_edges = np.histogram(power_select_list_1, bins=10)

            number_2, bin_edges = np.histogram(power_select_list_2, bins=10)

            number_3, bin_edges = np.histogram(power_select_list_3, bins=10)

            p_0 = number_0 / (number_0 + number_1 + number_2 + number_3)
            p_1 = number_1 / (number_0 + number_1 + number_2 + number_3)
            p_2 = number_2 / (number_0 + number_1 + number_2 + number_3)
            p_3 = number_3 / (number_0 + number_1 + number_2 + number_3)

            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_0, 'b*-', label='Power Level 23 dB')
            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_1, 'rs-', label='Power Level 10 dB')
            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_2, 'go-', label='Power Level 5 dB')
            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_3, 'y^-', label='Power Level 1 dB')
            plt.xlim([0, 0.12])
            plt.xlabel("Time left for V2V transmission (s)")
            plt.ylabel("Probability of power selection")
            plt.legend()
            plt.grid()
            plt.show()

            U2I_Rate_list[game_idx] = np.mean(np.asarray(U2IRate_list))
            U2U_Rate_list[game_idx] = np.mean(np.asarray(U2URate_list))
            success_percent_list[game_idx] = percent

            print('Mean of the U2I rate is that ', np.mean(U2I_Rate_list[0:game_idx]))
            print('Mean of the U2U rate is that ', np.mean(U2U_Rate_list[0:game_idx]))
            print('Mean of success percent is that ', percent, np.mean(success_percent_list[0:game_idx]))
            # print('Probability of satisfy is that ', probability_satisfied/(test_sample*len(self.env.UAVs)*3))
            # print('action is that', action_temp[0,:])

        print('The number of UAVs is ', len(self.env.UAVs))
        print('Mean of the U2I rate is that ', np.mean(U2I_Rate_list))
        print('Mean of the U2U rate is that ', np.mean(U2U_Rate_list))
        print('Mean of success percent is that ', np.mean(success_percent_list))
        # print('Test Reward is ', np.mean(test_result))
