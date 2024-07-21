from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import numpy as np

def main():
    a_env = StarCraft2Env(map_name="3s5z")
    env_info = a_env.get_env_info()
    
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    # n_episodes = 1
    a_env.reset()

    print('....... game: ', a_env.map_name ,'.......')
    sight = a_env.unit_sight_range(1)
    shoot_range1 = a_env.unit_shoot_range(1)
    shoot_range2 = a_env.unit_shoot_range(6)

    print('sight', sight)
    print('shoot_range 1', shoot_range1)
    print('shoot_range 2', shoot_range2)

    state = a_env.get_state()
    state_dict = a_env.get_state_dict()
    obs_agent_1 = a_env.get_obs_agent(1)
    # obs = a_env.get_obs()
    
    print('n_agents: ', n_agents)
    print('........... obs ...........')
    # print('obs_shape: ', (len(obs),obs[0].size))
    print('obs_agent 1_shape: ', obs_agent_1.size)
    print('move_feats_dim：',a_env.get_obs_move_feats_size())
    print('enemy_feats_dim：',a_env.get_obs_enemy_feats_size())
    print('ally_feats_dim：',a_env.get_obs_ally_feats_size())
    print('own_feats_dim：',a_env.get_obs_own_feats_size())

    print('........... state ...........')
    print('state_shape', state.size)
    print('state: ', state)
    print('state_allies_size', state_dict["allies"].size)  # 15
    # print('state_allies_k', state_dict["allies"][0,2],state_dict["allies"][0,3])
    print('state_allies_shape', state_dict["allies"].shape)
    print('state_enemies', state_dict["enemies"])
    print('state_enemies_shape', state_dict["enemies"].shape)
    print('last_action_k', state_dict["last_action"][0])
    print('last_action_shape', state_dict["last_action"].shape)


    print('--------------------------------------')
    print('--------------------------------------')
    a_env.get_obs_agent(1)
    print('agent_1_obs：','\n',a_env.get_obs_agent(1))
    # print('state','\n',state)



if __name__ == "__main__":
    main()
