# 测试版04 -- 在t2基础上+sr & 将局面三的reward改为仅与距离相关
# 与 p6_classing_scenario 对应

import numpy as np
import matplotlib.pyplot as plt
import torch

class RewardAllocation:
    def __init__(self, env,  batch , t, state_t, ):
        self.a_env = env
        self.env_info = self.a_env.get_env_info()
        self.batch = batch
        self.t = t
        self.n_agents = self.env_info["n_agents"]
        self.n_actions = self.env_info["n_actions"]
        self.n_gstate = self.a_env.get_state_dict()["allies"].size
        self.action_matrix = self.batch['actions'][:,self.t]
        self.reward_matrix = np.zeros((1,self.n_agents))
        self.state_matrix = self.batch['state']
        self.obs_matrix = self.batch['obs'][:,self.t]
        self.g_state_t0 = self.batch['state'][:,self.t,0:self.n_gstate] 
        self.g_state_t1 = state_t

    def scenario_reward(self,nt,nr,tar,sr):
        # self.fig_t()
        n = self.a_env.get_obs_move_feats_size() + np.prod(self.a_env.get_obs_enemy_feats_size())
        b1 = self.a_env.get_obs_ally_feats_size()[0]  # 几个 ally agents 
        b2 = self.a_env.get_obs_ally_feats_size()[1]  # 每个 ally agents的特征数量 
        
        # 修改视距范围
        for k in range(self.n_agents):
            obs_k = self.obs_matrix[:,k,:]
            self.sight_modification(obs_k,k,b1,b2,n)

        c_st = []
        for i in range(self.n_agents):
            obs_i = self.obs_matrix[:,i,:]
            state_all_zero = all(obs_i[:,n + b2 * j] == 0 for j in range(b1))
            c_st_i = 0 if state_all_zero else 1
            c_st.append(c_st_i)
                
        for k in range(self.n_agents):
            obs_k = self.obs_matrix[:,k,:]
            if all(c_st_i == 0 for c_st_i in c_st):
                lamda_id = 1
                lamda_1 = self.scenario_1_lamda(k) 
                # print(f"agent{k+1} 处于局面一",f"lamda_1: {lamda_1.item():.4f}")
                self.scenario_1_reward(k)
            elif all(c_st_i == 1 for c_st_i in c_st):
                lamda_id = 3
                lamda_3 = self.scenario_3_lamda(k,b1,b2,n)
                # print(f"agent{k+1} 处于局面三",f"lamda_3: {lamda_3.item():.4f}")
                self.scenario_3_reward(k)
            else:
                state_all_zero = all(obs_k[:,n + b2 * j] == 0 for j in range (b1))
                if state_all_zero:
                    lamda_id = 2
                    lamda_2 = self.scenario_2_lamda(k) 
                    # print(f"agent{k+1} 处于局面二", f"lamda_2: {lamda_2.item():.4f}")
                    self.scenario_2_reward(k)
                else:
                    lamda_id = 3
                    lamda_3 = self.scenario_3_lamda(k,b1,b2,n)
                    # print(f"agent{k+1} 处于局面三", f"lamda_3: {lamda_3.item():.4f}")
                    self.scenario_3_reward(k)
     
            nt,nr,tar,sr = self.tarcit(nt,nr,tar,k,lamda_id,sr)
        return self.reward_matrix,nt,nr,tar,sr

    def fig_t(self):
        map_x = self.a_env.max_distance_x
        sight_range = 9 / map_x
        fig, ax = plt.subplots(figsize=(10, 10))  
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')
        ticks = [i / 16 for i in range(-8, 8)]  # 生成从-0.5到0.5的刻度
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid(True)
        n_st = self.n_gstate/self.n_agents
        for i in range(self.n_agents):
            x = self.g_state_t1[int(n_st*i+2)]
            y = self.g_state_t1[int(n_st*i+3)]
            ax.plot(x, y, 'bo') 
            ax.text(x, y, f'Agent-{i+1} ({x:.2f}, {y:.2f})', ha='center', va='bottom')
            circle = plt.Circle((x, y), sight_range, color='r', linestyle='--', fill=False)
            ax.add_artist(circle)
        plt.savefig(f'/home/data_0/ysq_23/smac/work-1/src/results-fig/{self.t:02d}.png')
        plt.close()
    
    def sight_modification(self,obs_k,k,b1,b2,n):
        for j in range(b1):
            col_index = n + b2 * j + 1   
            target_col_index = n + b2 * j 
            print('obs_k: ',obs_k) 
            print('obs_k[:, col_index] : ', obs_k[:, col_index]) 
            mask = obs_k[:, col_index] > 0.5  
            obs_k[:, target_col_index][mask] = 0           
        self.obs_matrix[:,k,:] = obs_k     

    def scenario_1_lamda(self,k):
        min_distance = float('inf')
        n_st = self.n_gstate/self.n_agents
        x_k, y_k = self.g_state_t1[int(n_st * k + 2)], self.g_state_t1[int(n_st * k + 3)] 
        for j in range(self.n_agents):
            if j != k:
                x_j, y_j = self.g_state_t1[int(n_st * j + 2)], self.g_state_t1[int(n_st * j + 3)]
                distance = np.sqrt((x_k - x_j)**2 + (y_k - y_j)**2)
                if distance < min_distance:
                    min_distance = distance
        lamda_1 = min_distance
        return(lamda_1)

    def scenario_2_lamda(self,k):
        min_distance = float('inf')
        n_st = self.n_gstate/self.n_agents
        x_k, y_k = self.g_state_t1[int(n_st * k + 2)], self.g_state_t1[int(n_st * k + 3)]
        for j in range(self.n_agents):
            if j != k:
                x_j, y_j = self.g_state_t1[int(n_st * j + 2)], self.g_state_t1[int(n_st * j + 3)]
                distance = np.sqrt((x_k - x_j)**2 + (y_k - y_j)**2)
                if distance < min_distance:
                    min_distance = distance
        lamda_2 = min_distance
        return(lamda_2)

    def scenario_3_lamda(self,k,b1,b2,n):
        zero_matrix = np.zeros((b1), dtype=int)
        obs_k = self.obs_matrix[:,k,:]
        for j in range (b1):
            if obs_k[:,n + b2 * j] != 0:
                zero_matrix[j] = 1
        max_lamda_3 = -float('inf')
        for k2 in range(b1):
            if zero_matrix[k2] == 1:
                lamda_3_k2 = obs_k[:,n + b2 * k2 + 1]
                if lamda_3_k2 > max_lamda_3:
                    max_lamda_3 = lamda_3_k2
        lamda_3 = 1 - max_lamda_3
        return (lamda_3)

    def scenario_1_reward(self,k):
        n_st = self.n_gstate/self.n_agents
        r_agentk_t0 = np.sqrt((self.g_state_t0[:,int(n_st * k+2)])**2 + (self.g_state_t0[:,int(n_st * k+3)])**2)
        r_agentk_t1 = np.sqrt((self.g_state_t1[int(n_st * k+2)])**2 + (self.g_state_t1[int(n_st * k+3)])**2)
        r1 = r_agentk_t0 - r_agentk_t1
        # print(f'scenario_1_reward: {r1:.4f}')
        r1 = r1*8
        self.reward_matrix[:, k] = np.array([r1.item()], dtype=np.float32)
        return r1*8

    def scenario_2_reward(self,k):
        min_distance_0 = float('inf')
        min_distance = float('inf')
        n_st = self.n_gstate/self.n_agents
        x_k0, y_k0 = self.g_state_t0[:,int(n_st * k+2)], self.g_state_t0[:,int(n_st * k+3)]
        x_k, y_k = self.g_state_t1[int(n_st * k+2)], self.g_state_t1[int(n_st * k+3)] 
        for j in range(self.n_agents):
            if j != k:
                x_j, y_j = self.g_state_t0[:,int(n_st * j+2)], self.g_state_t0[:,int(n_st * j+3)]
                distance = np.sqrt((x_k0 - x_j)**2 + (y_k0 - y_j)**2)
                if distance < min_distance_0:
                    min_distance_0 = distance
                    min_distance = np.sqrt((x_k - x_j)**2 + (y_k - y_j)**2)
        r2 = min_distance_0 - min_distance
        # print(f'scenario_2_reward: {r2:.4f}')
        r2 = r2*8
        self.reward_matrix[:, k] = np.array([r2.item()], dtype=np.float32) 
        return r2*8

    def scenario_3_reward(self,k):
        obs_k_t1 = self.obs_matrix[:,k,:]
        n = self.a_env.get_obs_move_feats_size() + np.prod(self.a_env.get_obs_enemy_feats_size())
        b1 = self.a_env.get_obs_ally_feats_size()[0] 
        b2 = self.a_env.get_obs_ally_feats_size()[1]  
        n_st = self.n_gstate/self.n_agents
        max_distance_x = 32 
        sight_range = 9
        zero_matrix = np.zeros((b1), dtype=int)
        for j in range (b1):
            if obs_k_t1[:,n + b2 * j] != 0:
                zero_matrix[j] = 1
        max_lamda_30 = -float('inf')
        x_k0, y_k0 = self.g_state_t0[:,int(n_st * k+2)], self.g_state_t0[:,int(n_st * k+3)]
        x_k1, y_k1 = self.g_state_t1[int(n_st * k+2)], self.g_state_t1[int(n_st * k+3)]
        for k2 in range(b1):
            if zero_matrix[k2] == 1:
                k2 = k2 if k2 < k else k2 + 1
                al_x, al_y = self.g_state_t0[:,int(n_st * k2+2)], self.g_state_t0[:,int(n_st * k2+3)]
                lamda_3_k2 = np.sqrt((x_k0 - al_x)**2 + (y_k0 - al_y)**2)
                if lamda_3_k2 > max_lamda_30:
                    max_lamda_30 = lamda_3_k2 * np.sqrt(max_distance_x / sight_range)
                    lamda_3_kt = np.sqrt((x_k1 - al_x)**2 + (y_k1 - al_y)**2)
                    max_lamda_31 = lamda_3_kt * np.sqrt(max_distance_x / sight_range)
        
        max_lamda_30 = 1 if max_lamda_30 == float('-inf') else max_lamda_30
        max_lamda_31 = 1 - np.sqrt(1/sight_range) if max_lamda_30 == float('-inf') else max_lamda_31
        ra = max_lamda_30 - max_lamda_31
        r3 = 8 * ra
        self.reward_matrix[:, k] = np.array([r3.item()], dtype=np.float32)
        return r3

    def tarcit(self,nt,nr,tar,k,lamda_id,sr):
        nt[lamda_id-1] += 1
        if self.reward_matrix[:,k] != 0:
            sr[lamda_id-1] = (sr[lamda_id-1] + self.reward_matrix[:,k])/2
        if self.reward_matrix[:,k] > 0:
            nr[lamda_id-1] += 1
        tar[lamda_id-1] = nr[lamda_id-1]/nt[lamda_id-1]
        return nt,nr,tar,sr
