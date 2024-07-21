from sre_parse import State
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from components.t6_classing_scenario import RewardAllocation
import numpy as np


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.cur_tarcit = []
        self.cur_scenario_reward = []
        self.cur_tar_positive_reward = []
        self.cur_tar_total = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        
        # -------------------------------------------
        nr = np.zeros((4, 1))
        nt = np.zeros((4, 1))
        tar = np.zeros((4, 1))
        sr = np.zeros((4, 1))  # sr -- scenario-reward
        # -------------------------------------------

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            
            reward, terminated, env_info = self.env.step(actions[0])
            # ---------------------------------------------------------------------
            state_t = self.env.get_state() 
            reward_allocation = RewardAllocation(self.env, self.batch, self.t, state_t)
            reward,nt,nr,tar,sr = reward_allocation.scenario_reward(nt,nr,tar,sr)
            # ---------------------------------------------------------------------
            episode_return += reward
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        # ------------------------------------------------------
        # 改 batch 清洗数据
        self.batch.clean(self.env)
        # ------------------------------------------------------

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)
        self.cur_scenario_reward.append(sr)
        self.cur_tarcit.append(tar)
        self.cur_tar_positive_reward.append(nr)
        self.cur_tar_total.append(nt)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, self.cur_tarcit, self.cur_tar_positive_reward, self.cur_tar_total,self.cur_scenario_reward, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, self.cur_tarcit, self.cur_tar_positive_reward, self.cur_tar_total,self.cur_scenario_reward, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, tar, nr, nt, sr, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)

        self.logger.log_stat(prefix + "s1-reward-mean", np.mean([asr[0, 0] for asr in sr]), self.t_env)
        self.logger.log_stat(prefix + "s2-reward-mean", np.mean([asr[1, 0] for asr in sr]), self.t_env)
        self.logger.log_stat(prefix + "s3-reward-mean", np.mean([asr[2, 0] for asr in sr]), self.t_env)
        self.logger.log_stat(prefix + "s4-reward-mean", np.mean([asr[3, 0] for asr in sr]), self.t_env)

        self.logger.log_stat(prefix + "tar_positive_reward-1", np.mean([arr[0, 0] for arr in nr]), self.t_env)
        self.logger.log_stat(prefix + "tar_positive_reward-2", np.mean([arr[1, 0] for arr in nr]), self.t_env)
        self.logger.log_stat(prefix + "tar_positive_reward-3", np.mean([arr[2, 0] for arr in nr]), self.t_env)
        self.logger.log_stat(prefix + "tar_positive_reward-4", np.mean([arr[3, 0] for arr in nr]), self.t_env)

        self.logger.log_stat(prefix + "tar_total-1", np.mean([att[0, 0] for att in nt]), self.t_env)
        self.logger.log_stat(prefix + "tar_total-2", np.mean([att[1, 0] for att in nt]), self.t_env)
        self.logger.log_stat(prefix + "tar_total-3", np.mean([att[2, 0] for att in nt]), self.t_env)
        self.logger.log_stat(prefix + "tar_total-4", np.mean([att[3, 0] for att in nt]), self.t_env)

        self.logger.log_stat(prefix + "tarcit_1", np.mean([atar[0, 0] for atar in tar]), self.t_env)
        self.logger.log_stat(prefix + "tarcit_2", np.mean([atar[1, 0] for atar in tar]), self.t_env)
        self.logger.log_stat(prefix + "tarcit_3", np.mean([atar[2, 0] for atar in tar]), self.t_env)
        self.logger.log_stat(prefix + "tarcit_4", np.mean([atar[3, 0] for atar in tar]), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
