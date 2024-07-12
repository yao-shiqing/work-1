from smac.env import MultiAgentEnv, StarCraft2Env
import numpy as np
import math
import matplotlib.pyplot as plt

def main(map_name,iteration,):
    a_env = StarCraft2Env(map_name=map_name)
    env_info = a_env.get_env_info()
    n_agents = env_info["n_agents"]

    a_env.reset()

    while iter < iteration:
        state = a_env.get_state()
        obs = a_env.get_obs()
        # obs_k = a_env.get_obs_agent(k)

        iter = iter + 1


    '''
    while batch
        记录 obs、state、lamda矩阵
        用label求lamda
            引自 label.py -- class LamdaLabel -- 输入矩阵obs和矩阵state 输出矩阵lamda
        用model求lamda
            引自 model.py -- class SL_Network -- model(input) -- 输入矩阵obs 输出矩阵lamda
        记录二者的lamda
        随机改变下位置 -- def()
    求交叉熵-网络训练
    batch[9:1]-训练&测试
    '''


map_name = "3s5z"
main(map_name)
