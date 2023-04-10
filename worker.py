import os
from typing import Any

import numpy
import torch
from colorama import Fore
from redis import Redis
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.gamestates import PlayerData
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward
from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker

from CustomStateSetter import *
from obs.AstraObs import AstraObs

torch.set_num_threads(1)


class ExpandAdvancedObs(AstraObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)


def print_worker(data):
    print(Fore.CYAN + "Worker : ", end="")
    print(data)


class Worker:
    def __init__(self, team_size, obs_builder, action_parser, state_setter, rewards, rewards_weights,
                 terminal_conditions):
        self.team_size = team_size
        self.obs_builder = obs_builder
        self.action_parser = action_parser
        self.state_setter = state_setter
        self.rewards = rewards
        self.rewards_weights = rewards_weights
        self.terminal_conditions = terminal_conditions

    def match(self) -> Match:
        return Match(
            game_speed=100,
            spawn_opponents=True,
            team_size=3,
            state_setter=self.state_setter,
            obs_builder=self.obs_builder,
            action_parser=self.action_parser,
            terminal_conditions=self.terminal_conditions,
            reward_function=SB3CombinedLogReward(
                reward_functions=self.rewards,
                reward_weights=self.rewards_weights)
        )

    def run(self, redis, name):
        RedisRolloutWorker(redis, name, self.match(),
                           past_version_prob=.2,
                           evaluation_prob=0.01,
                           sigma_target=2,
                           dynamic_gm=True,
                           send_obs=True,
                           auto_minimize=False,
                           streamer_mode=False,
                           send_gamestates=False,
                           force_paging=False,
                           local_cache_name="Normal_astra_model_db").run()


if __name__ == "__main__":
    Worker(
        team_size=3,
        obs_builder=ExpandAdvancedObs(),
        action_parser=DiscreteAction(),
        state_setter=ProbabilisticStateSetter(),
        rewards=(),
        rewards_weights=(),
        terminal_conditions=[GoalScoredCondition(), TimeoutCondition(2000)]
    ).run(Redis(host="127.0.0.1", username="test-bot", password=os.environ["REDIS_PASSWORD"], port=6379,
                db=5), "Normal-astra")
