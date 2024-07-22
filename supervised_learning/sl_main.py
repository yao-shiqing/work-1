import torch
import numpy as np
import torch.nn as nn
import os
import torch.optim as optim
from smac.env import MultiAgentEnv, StarCraft2Env
from label import LamdaLabel
from model import SL_Network
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def main(map_name):
    a_env = StarCraft2Env(map_name=map_name)
    env_info = a_env.get_env_info()
    n_agents = env_info["n_agents"]
    n_tacit = 4
    lr = 0.00005
    time_steps = 500

    a_env.reset()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    a_current_time = current_time
    log_dir = os.path.join('/home/data_0/ysq_23/smac/work-1/src/supervised_learning/results',f'{current_time}','tb_logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

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

        # Training
        obs_matrix_tensor = torch.stack(obs_matrix)
        lamda_label_tensor = torch.stack(lamda_label_matrix)
        optimizer.zero_grad()
        lamda_net_tensor = sl_net.forward(obs_matrix_tensor)
        lamda_net_tensor = lamda_net_tensor.squeeze(1)
        loss = criterion(lamda_net_tensor, lamda_label_tensor)
        loss.backward()
        optimizer.step()
        writer.add_scalar('Training Loss', loss.item(), epoch)

        # Testing
        test_start = int(time_steps - 0.05*time_steps)
        test_lamda_label_tensor = torch.stack(lamda_label_matrix[test_start:])
        test_obs_matrix_tensor = torch.stack(obs_matrix[test_start:])
        test_output = sl_net(test_obs_matrix_tensor)
        test_loss = criterion(test_output, test_lamda_label_tensor)
        writer.add_scalar('Test Loss', test_loss.item(), epoch)

        # Log results
        with open(os.path.join('/home/data_0/ysq_23/smac/work-1/src/supervised_learning/results', f'{a_current_time}', 'training_log.txt'), 'a') as log_file:
            log_file.write(f"Epoch {epoch + 1}, Time: {current_time}, Training Loss: {loss.item()}, Test Loss: {test_loss.item()}\n")

        # Save model parameters
    model_path = os.path.join('/home/data_0/ysq_23/smac/work-1/src/supervised_learning/results', f'{a_current_time}', f'model_{epoch + 1}.pth')
    torch.save(sl_net.state_dict(), model_path)
    writer.close()

map_name = "3s_vs_5z_label"
main(map_name)
