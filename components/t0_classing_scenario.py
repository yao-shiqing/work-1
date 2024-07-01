# 测试版01
# 与 p6_classing_scenario 对应

from smac.env import StarCraft2Env
import numpy as np
import matplotlib.pyplot as plt

class RewardAllocation:
    def __init__(self, batch ,map_name , t, state_t):
        self.a_env = StarCraft2Env(map_name=map_name)
        self.env_info = self.a_env.get_env_info()
        self.batch = batch
        self.t = t
        self.nr = np.zeros((3, 1))
        self.nt = np.zeros((3, 1))
        self.tar = np.zeros((3, 1))
        self.n_agents = self.env_info["n_agents"]
        self.n_actions = self.env_info["n_actions"]
        self.n_gstate = self.a_env.get_state_dict()["allies"].size
        self.action_matrix = self.batch['actions'][:,self.t]
        self.reward_matrix = self.batch['rewards']
        self.state_matrix = self.batch['state']
        self.obs_matrix = self.batch['obs'][:,self.t]
        self.g_state_t0 = self.batch['state'][:,self.t,0:self.n_gstate] 
        self.g_state_t1 = state_t

    def scenario_reward(self):
        self.fig_t(self)
        n = self.a_env.get_obs_move_feats_size() + np.prod(self.a_env.get_obs_enemy_feats_size())
        b1 = self.a_env.get_obs_ally_feats_size()[0]  # 几个 ally agents 
        b2 = self.a_env.get_obs_ally_feats_size()[1]  # 每个 ally agents的特征数量    
        
        c_st = []
        for i in range(self.n_agents):
            obs_i = self.obs_matrix[:,i,:]
            state_all_zero = all(obs_i[n + b2 * j] == 0 for j in range(b1))
            c_st_i = 0 if state_all_zero else 1
            c_st.append(c_st_i)
                
        for k in range(self.n_agents):
            obs_k = self.obs_matrix[:,k,:]
            if all(c_st_i == 0 for c_st_i in c_st):
                lamda_id = 1
                lamda_1 = self.scenario_1_lamda(self,k) 
                print(f"agent{k+1} 处于局面一",f"lamda_1: {lamda_1:.4f}")
                self.scenario_1_reward(self,k)
            elif all(c_st_i == 1 for c_st_i in c_st):
                lamda_id = 3
                lamda_3 = self.scenario_3_lamda(self,k)
                print(f"agent{k+1} 处于局面三",f"lamda_3: {lamda_3:.4f}")
                self.scenario_3_reward(self,k)
            else:
                state_all_zero = all(obs_k[:,n + b2 * j] == 0 for j in range (b1))
                if state_all_zero:
                    lamda_id = 2
                    lamda_2 = self.scenario_2_lamda(self,k) 
                    print(f"agent{k+1} 处于局面二", f"lamda_2: {lamda_2:.4f}")
                    self.scenario_2_reward(self,k)
                else:
                    lamda_id = 3
                    lamda_3 = self.scenario_3_lamda(self,k)
                    print(f"agent{k+1} 处于局面三", f"lamda_3: {lamda_3:.4f}")
                    self.scenario_3_reward(self,k)
     
            self.tarcit(k,lamda_id)

    def fig_t(self):
        map_x = self.a_env.max_distance_x
        sight_range = 9 / map_x
        ax = plt.subplots(figsize=(10, 10))  
        ticks = [i / 16 for i in range(-8, 8)]  # 生成从-0.5到0.5的刻度
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid(True)
        n_st = self.n_gstate/self.n_agents
        for i in range(self.n_agents):
            x = self.g_state_t1[:,n_st*i+2]
            y = self.g_state_t1[:,n_st*i+3]
            ax.plot(x, y, 'bo') 
            ax.text(x, y, f'Agent-{i+1} ({x:.2f}, {y:.2f})', ha='center', va='bottom')
            circle = plt.Circle((x, y), sight_range, color='r', linestyle='--', fill=False)
            ax.add_artist(circle)
        plt.savefig(f'/home/data_0/ysq_23/smac/work-1/src/results-fig/{self.t:02d}.png')
        plt.close()

    def scenario_1_lamda(self,k):
        min_distance = float('inf')
        n_st = self.n_gstate/self.n_agents
        x_k, y_k = self.g_state_t1[:,n_st * k + 2], self.g_state_t1[:,n_st * k + 3] 
        for j in range(self.n_agents):
            if j != k:
                x_j, y_j = self.g_state_t1[:,n_st * j + 2], self.g_state_t1[:,n_st * j + 3]
                distance = np.sqrt((x_k - x_j)**2 + (y_k - y_j)**2)
                if distance < min_distance:
                    min_distance = distance
        lamda_1 = min_distance
        return(lamda_1)

    def scenario_2_lamda(self,k):
        min_distance = float('inf')
        n_st = self.n_gstate/self.n_agents
        x_k, y_k = self.g_state_t1[:,n_st * k + 2], self.g_state_t1[:,n_st * k + 3]
        for j in range(self.n_agents):
            if j != k:
                x_j, y_j = self.g_state_t1[:,n_st * j + 2], self.g_state_t1[:,n_st * j + 3]
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
        self.batch['state'][:,self.t,0:20]
        n_st = self.n_gstate/self.n_agents
        r_agentk_t0 = np.sqrt((self.g_state_t0[:,n_st * k+2])**2 + (self.g_state_t0[:,n_st * k+3])**2)
        r_agentk_t1 = np.sqrt((self.g_state_t1[:,n_st * k+2])**2 + (self.g_state_t1[:,n_st * k+3])**2)
        r1 = r_agentk_t0 - r_agentk_t1
        print(f'scenario_1_reward: {r1:.4f}')
        self.reward_matrix[:,k] = r1
        return r1*8

    def scenario_2_reward(self,k):
        min_distance_0 = float('inf')
        min_distance = float('inf')
        n_st = self.n_gstate/self.n_agents
        x_k0, y_k0 = self.g_state_t0[:,n_st * k+2], self.g_state_t0[:,n_st * k+3]
        for j in range(self.n_agents):
            if j != k:
                x_j, y_j = self.g_state_t0[:,n_st * j+2], self.g_state_t0[:,n_st * j+3]
                distance = np.sqrt((x_k0 - x_j)**2 + (y_k0 - y_j)**2)
                if distance < min_distance_0:
                    min_distance_0 = distance
        x_k, y_k = self.g_state_t1[:,n_st * j+2], self.g_state_t1[:,n_st * j+3] 
        for j in range(self.n_agents):
            if j != k:
                x_j, y_j = self.g_state_t0[:,n_st * j+2], self.g_state_t0[:,n_st * j+3]
                distance = np.sqrt((x_k - x_j)**2 + (y_k - y_j)**2)
                if distance < min_distance:
                    min_distance = distance
        r2 = min_distance_0 - min_distance
        print(f'scenario_2_reward: {r2:.4f}')
        self.reward_matrix[k,:] = r2 
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
            if obs_k_t1[n + b2 * j] != 0:
                zero_matrix[j] = 1
        max_lamda_30 = -float('inf')
        x_k0, y_k0 = self.g_state_t0[:,n_st * k+2], self.g_state_t0[:,n_st * k+3]
        x_k1, y_k1 = self.g_state_t1[:,n_st * k+2], self.g_state_t1[:,n_st * k+3]
        for k2 in range(b1):
            if zero_matrix[k2] == 1:
                k2 = k2 if k2 < k else k2 + 1
                al_x, al_y = self.g_state_t0[:,n_st * k2+2], self.g_state_t0[:,n_st * k2+3]
                lamda_3_k2 = np.sqrt((x_k0 - al_x)**2 + (y_k0 - al_y)**2)
                if lamda_3_k2 > max_lamda_30:
                    max_lamda_30 = lamda_3_k2 * np.sqrt(max_distance_x / sight_range)
                    lamda_3_kt = np.sqrt((x_k1 - al_x)**2 + (y_k1 - al_y)**2)
                    max_lamda_31 = lamda_3_kt * np.sqrt(max_distance_x / sight_range)
        
        max_lamda_30 = 1 if max_lamda_30 == float('-inf') else max_lamda_30
        max_lamda_31 = 1 - np.sqrt(1/sight_range) if max_lamda_30 == float('-inf') else max_lamda_31
        ra = max_lamda_30 - max_lamda_31

        directions = {
        1: (0,0),
        2: (0, 1),
        3: (0, -1),
        4: (1, 0),
        5: (-1, 0)}
        ct = directions.get(self.action_matrix[:, k, :].item(), (0, 0))
        cx,cy = self.g_state_t0[:,n_st * k+2], self.g_state_t0[:,n_st * k+3]
        if cy > 0 and cx > 0:
            dx, dy = 0.5-cx, 0.5-cy
            ce = (0, -1) if dx > dy else (-1, 0)
        elif cy < 0 and cx > 0:
            dx, dy = 0.5-cx, cy + 0.5
            ce = (0, 1) if dx > dy else (-1, 0)
        elif cy > 0 and cx < 0:
            dx, dy = 0.5 + cx, 0.5 - cy
            ce = (0, -1) if dx > dy else (1, 0)   
        elif cy < 0 and cx < 0:
            dx, dy = 0.5 + cx, cy + 0.5
            ce = (0, 1) if dx > dy else (1, 0)  
        else:
            ce = (0,0)
        print('c_exp：',ce,'; c_t：',ct,)
        c1 = np.linalg.norm(np.array(ct) - np.array(ce))
        rb = (0.5 - c1)/3
        miu_a = 1- self.scenario_3_lamda(self,k,b1,b2,n)
        miu_b = self.scenario_3_lamda(self,k,b1,b2,n)
        r3 = miu_a * ra + miu_b * rb
        ma = miu_a * ra 
        mb = miu_b * rb
        print(f'scenario_3_reward: {r3:.4f}; ra:{ra:.4f}; rb:{rb:.4f}; miu_a*ra:{ma:.4f};  miu_b*rb:{mb:.4f}')
        self.reward_matrix[k,:] = r3 
        return r3

    def tarcit(self,k,lamda_id):
        self.nt[lamda_id] += 1
        if self.reward_matrix[k,:] > 0:
                self.nr[lamda_id] += 1
                self.tar[lamda_id] = self.nr[lamda_id]/self.nt[lamda_id]
