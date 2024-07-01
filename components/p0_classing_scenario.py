# 初稿01

from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import numpy as np

def estimate_scenario(k, map_name):
    a_env = StarCraft2Env(map_name=map_name)
    env_info = a_env.get_env_info()
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    a_env.reset()
    state = a_env.get_state()
    state_dict = a_env.get_state_dict()
    state_allies = state_dict["allies"]
    last_action = state_dict["last_action"]

    c_st = []
    n = a_env.get_obs_move_feats_size() + np.prod(a_env.get_obs_enemy_feats_size())
    b1 = a_env.get_obs_ally_feats_size()[0]  # 几个 ally agents 
    b2 = a_env.get_obs_ally_feats_size()[1]  # 每个 ally agents的特征数量

    for i in range(n_agents):
        obs_agent_i = a_env.get_obs_agent(i)
        state_all_zero = all(obs_agent_i[n + b2 * j] == 0 for j in range(b1-1))
        c_st_i = 0 if state_all_zero else 1
        c_st.append(c_st_i)

    print('.......................................')
    if all(c_st_i == 0 for c_st_i in c_st):
        print(f"agent_{k+1} 处于局面一")
        scenario_1_lamda(k, state_dict)  
    elif all(c_st_i == 1 for c_st_i in c_st):
        print(f"agent_{k+1} 处于局面三")
        scenario_3_lamda(a_env, k, n, b1, b2)
    else:
        obs_agent_k = a_env.get_obs_agent(k)
        state_all_zero = all(obs_agent_k[n + b2 * j] == 0 for j in range(b1-1))
        if state_all_zero:
            print(f"agent_{k+1} 处于局面二")
            scenario_2_lamda(k, state_dict) 
        else:
            print(f"agent_{k+1} 处于局面三")
            scenario_3_lamda(a_env, k, n, b1, b2)

def scenario_1_lamda(k,state_dict):
    g_state = state_dict["allies"]
    min_distance = float('inf')
    num_allies = g_state.shape[0]
    x_k, y_k = g_state[k, 2], g_state[k, 3] 
    for j in range(num_allies):
        if j != k:
            x_j, y_j = g_state[j, 2], g_state[j, 3]
            distance = np.sqrt((x_k - x_j)**2 + (y_k - y_j)**2)
            if distance < min_distance:
                min_distance = distance
    lamda_1 = min_distance
    print("lamda_1:", lamda_1)
    return(lamda_1)

def scenario_2_lamda(k,state_dict):
    g_state = state_dict["allies"]
    min_distance = float('inf')
    num_allies = g_state.shape[0]
    x_k, y_k = g_state[k, 2], g_state[k, 3] 
    for j in range(num_allies):
        if j != k:
            x_j, y_j = g_state[j, 2], g_state[j, 3]
            distance = np.sqrt((x_k - x_j)**2 + (y_k - y_j)**2)
            if distance < min_distance:
                min_distance = distance
    lamda_2 = min_distance
    print("lamda_2:", lamda_2)
    return(lamda_2)

def scenario_3_lamda(a_env, k, n, b1, b2):
    zero_matrix = np.zeros((b1), dtype=int)
    
    obs_agent_k = a_env.get_obs_agent(k)
    for j in range(b1 - 1):
        if obs_agent_k[n + b2 * j] != 0:
            zero_matrix[j] = 1

    max_lamda_3 = -float('inf')
    for k2 in range(b1 - 1):
        if zero_matrix[k2] == 1:
            lamda_3_k2 = a_env.get_obs_agent(k)[n + b2 * k2 + 1]
            if lamda_3_k2 > max_lamda_3:
                max_lamda_3 = lamda_3_k2
    lamda_3 = 1 - max_lamda_3
    print("lamda_3:", lamda_3)
    return (lamda_3)

'''
def scenario_1_reward(k,state_allies):
    state_agent_k = np.sqrt((state_allies[k,2])**2 + (state_allies[k,3])**2)
    r1 = r(t-1) - state_agent_k
    return r1

def scenario_2_reward(k,state_dict):
    min_distance = float('inf')
    num_allies = g_state.shape[0]
    x_k, y_k = g_state[k, 2], g_state[k, 3] 
    for j in range(num_allies):
        if j != k:
            x_j, y_j = g_state[j, 2], g_state[j, 3]
            distance = np.sqrt((x_k - x_j)**2 + (y_k - y_j)**2)
            if distance < min_distance:
                min_distance = distance
    r2 = r(t-1) - min_distance
    return r2

def scenario_3_reward(k,lamda_3,last_action,state_allies):
    ra = lamda_3(t) - lamda_3(t-1)  # 通过st-(t-1)计算

    if last_action[k][2] == 1:
        ct = (0, 1)
    elif last_action[k][3] == 1:
        ct = (0, -1)
    elif last_action[k][4] == 1:
        ct = (1, 0)
    elif last_action[k][5] == 1:
        ct = (-1, 0)
    else:
        return (0, 0)  
    
    cx = state_allies[k][2]
    cy = state_allies[k][3]

    if cy > 0 and abs(cx) < cy:
        ce = (0, 1)
    elif cy < 0 and abs(cx) < abs(cy):
        ce = (0, -1)
    elif cx > 0 and abs(cy) < cx:
        ce = (1, 0)
    elif cx < 0 and abs(cy) < abs(cx):
        ce = (-1, 0)
    else:
        return (0, 0)  

    c = np.linalg.norm(np.array(ct) - np.array(ce))
    rb = 1 - c/5
    miu_a = 0.5
    miu_b = 0.5
    r3 = miu_a * ra + miu_b * rb
    return r3

'''


k = 0 
map_name = "3m"

estimate_scenario(k, map_name)

