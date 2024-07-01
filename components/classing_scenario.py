from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import numpy as np

def main():
    a_env = StarCraft2Env(map_name="3s_vs_5z")
    env_info = a_env.get_env_info()
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    a_env.reset()

    state = a_env.get_state()
    state_dict = a_env.get_state_dict()

    # scenario 1 -- green points
    i = n_agents - 1 
    obs_agent_i = a_env.get_obs_agent(i)
    n = a_env.get_obs_move_feats_size() + np.prod(a_env.get_obs_enemy_feats_size())
    a = obs_agent_i[n]
    b1 = a_env.get_obs_ally_feats_size()[0]  # 几个 ally agents 
    b2 = a_env.get_obs_ally_feats_size()[1]  # 每个 ally agents的特征数量
    for j in range(n_agents-1):
        obs_agent_j = a_env.get_obs_agent(j)
        scen1 = any(obs_agent_j[n + b1 * j] == 1 for j in range(b2-1))  # 有1则为True
        if scen1:
            ifnot_scen1 = True
            print('agenti 不属于局面一')
            break

    if ifnot_scen1:
        lamda_1 = 0
    else:
        print('agenti 属于局面一')
        g_state = state_dict["allies"]
        min_distance = float('inf')
        num_allies = g_state.shape[0]
        x_i, y_i = g_state[i, 2], g_state[i, 3] 
        for j in range(num_allies):
            if j != i:
                x_j, y_j = g_state[j, 2], g_state[j, 3]
                distance = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
            if distance < min_distance:
                min_distance = distance
        lamda_1 = min_distance
    
    print("lamda_1:", lamda_1)

    
    print('--------------------------------------')


    # scenario 2 -- blue points


    # scenario 3 -- red points


if __name__ == "__main__":
    main()


'''
 # scenario 1 -- green points

    obs_agent_i = a_env.get_obs_agent(i)
    n = a_env.get_obs_move_feats_size() + np.prod(a_env.get_obs_enemy_feats_size())
    a = obs_agent_i[n]
    b1 = a_env.get_obs_ally_feats_size()[0]  # 几个 ally agents 
    b2 = a_env.get_obs_ally_feats_size()[1]  # 每个 ally agents的特征数量
    for j in range(n_agents-1):
        obs_agent_j = a_env.get_obs_agent(j)
        scen1 = any(obs_agent_j[n + b1 * j] == 1 for j in range(b2-1))  # 有1则为True
        if scen1:
            ifnot_scen1 = True
            print('agenti 不属于局面一')
            break

def scenario_1_lamda(i):
    g_state = state_dict["allies"]
    min_distance = float('inf')
    num_allies = g_state.shape[0]
    x_i, y_i = g_state[i, 2], g_state[i, 3] 
    for j in range(num_allies):
        if j != i:
            x_j, y_j = g_state[j, 2], g_state[j, 3]
            distance = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
        if distance < min_distance:
            min_distance = distance
    lamda_1 = min_distance
    print("lamda_1:", lamda_1)
    return(lamda_1)
'''