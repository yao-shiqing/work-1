from smac.env import MultiAgentEnv, StarCraft2Env
import numpy as np
import math
import matplotlib.pyplot as plt

def main(map_name,iteration,):
    a_env = StarCraft2Env(map_name=map_name)
    env_info = a_env.get_env_info()
    n_agents = env_info["n_agents"]
<<<<<<< HEAD
=======
    n_tacit = 4
    lr = 0.00005
    time_steps = 500
>>>>>>> 2ac4495 (日期0721 修改了局面三&局面四的lamda和reward 使其得以适应不同视距的智能体 同时做了数据清洗 去除了具有敌方智能体的数据)

    a_env.reset()

<<<<<<< HEAD
    while iter < iteration:
        state = a_env.get_state()
        obs = a_env.get_obs()
        # obs_k = a_env.get_obs_agent(k)
=======
    for epoch in range(1000):
        obs = a_env.get_obs()
        n_obs = len(a_env.get_obs()[0])
        lamda_label_matrix = []
        obs_matrix = []

        sl_net = SL_Network(n_agents, n_tacit, n_obs)
        optimizer = optim.Adam(sl_net.parameters(), lr)
        criterion = nn.SmoothL1Loss()


        for time in range(time_steps):
            actions = []
            for agent_id in range(n_agents):
                avail_actions = a_env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)
            a_env.step(actions)

            lamda_rule = LamdaLabel(a_env)
            lamda_label = lamda_rule.scenario_lamda()
            lamda_label_tensor = [torch.from_numpy(arr) for arr in lamda_label]
            lamda_label_stacked_tensor = torch.stack(lamda_label_tensor)

            obs = a_env.get_obs()
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            input_sample = obs_tensor.unsqueeze(0)

            obs_matrix.append(input_sample)
            lamda_label_matrix.append(lamda_label_stacked_tensor)
>>>>>>> 2ac4495 (日期0721 修改了局面三&局面四的lamda和reward 使其得以适应不同视距的智能体 同时做了数据清洗 去除了具有敌方智能体的数据)

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


<<<<<<< HEAD
map_name = "3s5z"
=======
map_name = "3s_vs_5z_label"
>>>>>>> 2ac4495 (日期0721 修改了局面三&局面四的lamda和reward 使其得以适应不同视距的智能体 同时做了数据清洗 去除了具有敌方智能体的数据)
main(map_name)
