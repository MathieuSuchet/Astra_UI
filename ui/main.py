import json
import random
from multiprocessing import Process
from typing import Union, Any, List

import numpy as np
import requests
import rlgym.utils.common_values as constants
from rlgym.envs import Match
from rlgym.utils.gamestates import PlayerData, PhysicsObject, GameState
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward, FaceBallReward, \
    TouchBallReward

from Rewards import UITraceableReward


def send_json_to_server(dest, json):
    return requests.get(f"http://127.0.0.1:5000/{dest}", json=json,
                        headers={
                            "Content-Type": 'application/json'
                        })


class PlayerDataJsonEncoder(json.JSONEncoder):
    def encode(self, o: Any) -> str:
        if isinstance(o, type(PlayerData())):
            return "{" \
                   f"\"car_id\": {json.JSONEncoder().encode(o.car_id)}," \
                   f"\"team_num\": {json.JSONEncoder().encode(o.team_num)}," \
                   f"\"boost_amount\": {json.JSONEncoder().encode(o.boost_amount)}," \
                   f"\"car_data\": {PhysicsObjectJsonEncoder().encode(o.car_data)}" \
                   "}"

        if isinstance(o, type([])):
            if len(o) > 0 and isinstance(o[0], type(PlayerData())):
                jsontext = "{\"data\": {" \
                           f"\"length\": {len(o)}," \
                           f"\"players\": ["
                for player in o:
                    jsontext += PlayerDataJsonEncoder().encode(player)
                    if o.index(player) == len(o) - 1:
                        continue

                    jsontext += ","
                jsontext += "]}}"
                return jsontext


class PhysicsObjectJsonEncoder(json.JSONEncoder):
    def encode(self, o: Any) -> str:
        if isinstance(o, type(PhysicsObject())):
            return "{" \
                   f"\"position\": {json.JSONEncoder().encode(o.position.tolist())}," \
                   f"\"linear_velocity\": {json.JSONEncoder().encode(o.linear_velocity.tolist())}," \
                   f"\"angular_velocity\": {json.JSONEncoder().encode(o.angular_velocity.tolist())}," \
                   f"\"forward\": {json.JSONEncoder().encode(o.forward().tolist())}," \
                   f"\"up\": {json.JSONEncoder().encode(o.up().tolist())}" \
                   "}"


class Software(Process):
    PORT = 6398
    HOST = "127.0.0.1"

    def run_ui(self):
        p1 = PlayerData()
        p1.car_id = 1
        p1.team_num = constants.BLUE_TEAM
        p1.car_data = PhysicsObject(position=np.array(
            [random.randint(-500, 500), random.randint(-500, 500), random.randint(0, 500)]),
            quaternion=np.array([0, 1, 1, 0]))

        p2 = PlayerData()
        p2.car_id = 2
        p2.team_num = constants.ORANGE_TEAM
        p2.car_data = PhysicsObject(position=np.array([-500.84546854, -1020.8746544651, 60.56645456456]),
                                    quaternion=np.array([0, 0, 1, 0.0000463]),
                                    linear_velocity=np.array([45, 45, 0]))

        p3 = PlayerData()
        p3.car_id = 3
        p3.team_num = constants.BLUE_TEAM
        p3.car_data = PhysicsObject(position=np.array([500, -1020, 60]),
                                    quaternion=np.array([1, 0, 0, 0]))

        p4 = PlayerData()
        p4.car_id = 4
        p4.team_num = constants.ORANGE_TEAM
        p4.car_data = PhysicsObject(position=np.array(
            [random.randint(-500, 500), random.randint(-500, 500), random.randint(0, 500)]),
            quaternion=np.array([0, 1, 1, 0]))

        p5 = PlayerData()
        p5.car_id = 5
        p5.team_num = constants.ORANGE_TEAM
        p5.car_data = PhysicsObject(position=np.array([-500.84546854, -1020.8746544651, 60.56645456456]),
                                    quaternion=np.array([0, 0, 1, 0.0000463]),
                                    linear_velocity=np.array([45, 45, 0]))

        p6 = PlayerData()
        p6.car_id = 6
        p6.team_num = constants.BLUE_TEAM
        p6.car_data = PhysicsObject(position=np.array([500, -1020, 60]),
                                    quaternion=np.array([1, 0, 0, 0]))

        players = [
            p1, p2, p3, p4, p5, p6
        ]

        send_json_to_server("bot_connection", json.JSONEncoder().encode(True))
        send_json_to_server("agents_summaries", PlayerDataJsonEncoder().encode(players))

        g = GameState()
        g.ball.position = [100, 400, 200]
        g.players = players

        previous_action = np.array([-0.5, 0.5, 1, 1, 1, 1, 1, 1])

        UITraceableReward.UITraceableReward(
            reward_functions=(
                VelocityBallToGoalReward(),
                # EventReward(
                #     goal=100,
                #     concede=-100,
                #     save=50,
                # ),
                FaceBallReward(),
                TouchBallReward(),
            ),
            reward_weights=(1.0, 1.0, 1.0)
        ).get_final_reward(p2, g, previous_action)

        raise Exception("Stopping")


class AstraMatch(Match):
    HOST = "127.0.0.1"
    PORT = 19006

    def __init__(self,
                 reward_function,
                 terminal_conditions,
                 obs_builder,
                 action_parser,
                 state_setter,
                 team_size=1,
                 tick_skip=8,
                 game_speed=100,
                 gravity=1,
                 boost_consumption=1,
                 spawn_opponents=False,
                 ):
        super().__init__(reward_function, terminal_conditions, obs_builder, action_parser, state_setter, team_size,
                         tick_skip, game_speed, gravity, boost_consumption, spawn_opponents)
        self.package_wait = 8

    def build_observations(self, state) -> Union[Any, List]:
        obs = super(AstraMatch, self).build_observations(state)

        self.package_wait -= 1
        if self.package_wait == 0:


            self.package_wait = 8
        return obs


s = Software()

s.run_ui()
requests.get("http://127.0.0.1:5000/bot_connection", json=json.JSONEncoder().encode(False), headers={
    "Content-Type": 'application/json'
})
