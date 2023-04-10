import os
from typing import Any, Type, Union, Tuple, List

import numpy
import torch.jit
from colorama import Fore
from redis import Redis
from rlgym.utils import RewardFunction, ObsBuilder
from rlgym.utils.action_parsers import ActionParser
from rlgym_compat import PlayerData, GameState
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward
from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.stat_trackers.stat_tracker import StatTracker
from rocket_learn.utils.util import SplitLayer
from torch import Tensor
from torch.nn import Linear, Sequential, Tanh

import wandb
from obs.AstraObs import AstraObs


def print_learner(data):
    print(Fore.YELLOW + "Learner : ", end="")
    print(data)


def create_agent():
    # ROCKET-LEARN EXPECTS A SET OF DISTRIBUTIONS FOR EACH ACTION FROM THE NETWORK, NOT
    # THE ACTIONS THEMSELVES. SEE network_setup.readme.txt FOR MORE INFORMATION
    split = (3, 3, 3, 3, 3, 2, 2, 2)
    total_output = sum(split)

    # TOTAL SIZE OF THE INPUT DATA
    state_dim = 231

    critic = Sequential(
        Linear(state_dim, 512),
        Tanh(),
        Linear(512, 512),
        Tanh(),
        Linear(512, 256),
        Tanh(),
        Linear(256, 256),
        Tanh(),
        Linear(256, 256),
        Tanh(),
        Linear(256, 1)
    )

    actor = DiscretePolicy(Sequential(
        Linear(state_dim, 512),
        Tanh(),
        Linear(512, 512),
        Tanh(),
        Linear(512, 256),
        Tanh(),
        Linear(256, 256),
        Tanh(),
        Linear(256, 256),
        Tanh(),
        Linear(256, total_output),
        SplitLayer(splits=split)
    ), split)

    # CREATE THE OPTIMIZER
    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": 5e-5},
        {"params": critic.parameters(), "lr": 5e-5}
    ])

    # PPO REQUIRES AN ACTOR/CRITIC AGENT
    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)
    state_dict = agent.state_dict()

    params: Tensor = torch.load("exit_save/policy.pth", map_location=torch.device("cpu"))

    new_params = torch.cat((params["mlp_extractor.shared_net.0.weight"], torch.rand([512, 62]),
                            params["mlp_extractor.shared_net.0.weight"][:, 76:107].repeat(1, 2)), dim=1)

    state_dict["actor.net.0.weight"] = new_params
    state_dict["critic.0.weight"] = new_params

    state_dict["actor.net.0.bias"] = params["mlp_extractor.shared_net.0.bias"]
    state_dict["critic.0.bias"] = params["mlp_extractor.shared_net.0.bias"]

    state_dict["actor.net.2.weight"] = params["mlp_extractor.shared_net.2.weight"]
    state_dict["critic.2.weight"] = params["mlp_extractor.shared_net.2.weight"]

    state_dict["actor.net.2.bias"] = params["mlp_extractor.shared_net.2.bias"]
    state_dict["critic.2.bias"] = params["mlp_extractor.shared_net.2.bias"]

    state_dict["actor.net.4.weight"] = params["mlp_extractor.policy_net.0.weight"]
    state_dict["critic.4.weight"] = params["mlp_extractor.value_net.0.weight"]

    state_dict["actor.net.4.bias"] = params["mlp_extractor.policy_net.0.bias"]
    state_dict["critic.4.bias"] = params["mlp_extractor.value_net.0.bias"]

    state_dict["actor.net.6.weight"] = params["mlp_extractor.policy_net.2.weight"]
    state_dict["critic.6.weight"] = params["mlp_extractor.value_net.2.weight"]

    state_dict["actor.net.6.bias"] = params["mlp_extractor.policy_net.2.bias"]
    state_dict["critic.6.bias"] = params["mlp_extractor.value_net.2.bias"]

    state_dict["actor.net.8.weight"] = params["mlp_extractor.policy_net.4.weight"]
    state_dict["critic.8.weight"] = params["mlp_extractor.value_net.4.weight"]

    state_dict["actor.net.8.bias"] = params["mlp_extractor.policy_net.4.bias"]
    state_dict["critic.8.bias"] = params["mlp_extractor.value_net.4.bias"]

    state_dict["actor.net.10.weight"] = params["action_net.weight"]
    state_dict["critic.10.weight"] = params["value_net.weight"]

    state_dict["actor.net.10.bias"] = params["action_net.bias"]
    state_dict["critic.10.bias"] = params["value_net.bias"]

    agent.load_state_dict(state_dict)
    return agent


class Learner:
    def __init__(
            self,
            obs_builder: Type[ObsBuilder],
            action_parser: Type[ActionParser],
            rewards: Union[Tuple[Type[RewardFunction]], List[Type[RewardFunction]]],
            rewards_weights: Union[Tuple[float], List[float]],
            stats_trackers: Union[Tuple[StatTracker], List[StatTracker]]
    ):
        self.stats_trackers = stats_trackers
        self.rewards_weights = rewards_weights
        self.rewards = rewards
        self.agent = create_agent()

        if len(rewards_weights) != len(rewards):
            raise Exception(
                f"Rewards weight and rewards have different lengths, expecting both to have same length : \n"
                f"Rewards length : {len(rewards)}\n"
                f"Rewards weights length : {len(rewards_weights)}")

        self.action_parser = action_parser
        self.obs_builder = obs_builder

    def run(self, logger_name, redis_logger_name: str = ""):
        wandb.login(key=os.environ["WANDB_KEY"])
        logger = wandb.init(project="demo", entity="cryy_salt")
        logger.name = logger_name

        if redis_logger_name == "":
            redis_logger_name = logger_name

        redis = Redis(host="127.0.0.1", password=os.environ["REDIS_PASSWORD"], username="test-bot", port=6379, db=5)

        def obs():
            return self.obs_builder()

        def rew():
            return SB3CombinedLogReward(
                reward_functions=tuple(reward() for reward in self.rewards),
                reward_weights=self.rewards_weights
            )

        def act():
            return self.action_parser()

        rollout_gen = RedisRolloutGenerator(redis_logger_name, redis, obs, rew, act,
                                            logger=logger,
                                            save_every=100,
                                            model_every=100,
                                            stat_trackers=self.stats_trackers,
                                            clear=False)

        # INSTANTIATE THE PPO TRAINING ALGORITHM
        alg = PPO(
            rollout_gen,
            self.agent,
            ent_coef=0.01,
            n_steps=100_000,
            batch_size=50_000,
            minibatch_size=10_000,
            epochs=10,
            gamma=599 / 600,
            clip_range=0.2,
            gae_lambda=0.95,
            vf_coef=1,
            max_grad_norm=0.5,
            logger=logger,
            device="cpu",
        )

        # BEGIN TRAINING. IT WILL CONTINUE UNTIL MANUALLY STOPPED
        # -iterations_per_save SPECIFIES HOW OFTEN CHECKPOINTS ARE SAVED
        # -save_dir SPECIFIES WHERE
        alg.run(iterations_per_save=100, save_dir="checkpoint_save_directory")


class ExpandAdvancedObs(AstraObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)


if __name__ == "__main__":
    print("We don't do that here, execute Astra.py file")
