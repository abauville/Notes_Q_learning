#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 15:37:03 2021

@author: abauville

Frozen lake environment with a random agent

Exercise for lesson 2 of Modern Reinforcement Learning: Deep Q Learning in PyTorch
https://www.udemy.com/course/deep-q-learning-from-paper-to-code/learn/lecture/17009504#overview


"""



import gym
import numpy as np
import matplotlib.pyplot as plt

#There are 16 possible states, i.e. 1 per square on the grid
# Here is the grid
# SFFF 0123
# FHFH 4567
# FFFH ...
# HFFG

# Possible actions: 0=left, 1=down, 2=right, 3=up, -1=NA

policy = {
    0  : 1,
    1  : 2,
    2  : 1,
    3  : 0,
    4  : 1,
    5  : -1,
    6  : 1,
    7  : -1,
    8  : 2,
    9  : 1,
    10 : 1,
    11 : -1,
    12 : -1,
    13 : 2,
    14 : 2,
    15 : -1}


n_episode = 1000

env = gym.make('FrozenLake-v0',  is_slippery = True  )
num_wins = 0.0
n_window = 10 # num of episodes to record
reward_list = []
for i_episode in range(n_episode):
    observation = env.reset()
    for t in range(10000):
        # env.render()
        # print(observation)
        # action = env.action_space.sample()
        action = policy[observation]
        observation, reward, done, info = env.step(action)
        if done:
            reward_list.append(reward)
            print(f"Episode {i_episode:03d} finished after {t+1} timesteps")
            break

env.close()

reward_list = np.array(reward_list)
win_ratio = np.array([np.mean(reward_list[i:i+n_window])
                      for i in range(1,len(reward_list), n_window)])


plt.clf()
plt.plot(win_ratio)
# print(f"winning ratio after 1000 episodes: {win_ratio:.5f}")


