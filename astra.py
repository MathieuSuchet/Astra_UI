import json
import os
import sys
import time
from io import StringIO
from multiprocessing import Pipe, Process
from threading import Thread, Event
from typing import Any

import numpy
from colorama import Fore
from redis import Redis
from rlgym.utils.gamestates import PlayerData, GameState
from typing_extensions import T

from learner import Learner, print_learner
from obs.AstraObs import AstraObs
from worker import Worker, print_worker

import threading
import ctypes
import time


class Capturing(list):
    def __init__(self, ui_pipe):
        super().__init__()
        self.ui_pipe = ui_pipe

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio_output = StringIO()
        self._stderr = sys.stderr
        sys.stderr = self._stringio_output
        return self

    def __exit__(self, *args):
        self.extend(self._stringio_output.getvalue().splitlines())
        del self._stringio_output
        sys.stdout = self._stdout
        sys.stderr = self._stderr


class WorkerCapturing(Capturing):
    def append(self, __object: T) -> None:
        super(WorkerCapturing, self).append(__object)
        self.ui_pipe.send({
            "type": "wOutput",
            "data": __object
        })


class LearnerCapturing(Capturing):
    def append(self, __object: T) -> None:
        super(LearnerCapturing, self).append(__object)
        self.ui_pipe.send({
            "type": "lOutput",
            "data": __object
        })


class Astra:
    def __init__(
            self,
            obs_builder,
            action_parser,
            state_setter,
            rewards,
            rewards_weights,
            terminal_conditions,
            ui_pipe: Pipe
    ):
        self.learner_capturer = Capturing(ui_pipe)
        self.worker_capturer = Capturing(ui_pipe)
        self.rewards_weights = rewards_weights
        self.rewards = rewards
        self.state_setter = state_setter
        self.action_parser = action_parser
        self.obs_builder = obs_builder
        self.learner = Learner(
            obs_builder=obs_builder,
            action_parser=action_parser,
            rewards=rewards,
            rewards_weights=rewards_weights,
            stats_trackers=[]
        )
        self.worker = Worker(
            team_size=3,
            obs_builder=obs_builder,
            action_parser=action_parser,
            state_setter=state_setter,
            rewards=rewards,
            rewards_weights=rewards_weights,
            terminal_conditions=terminal_conditions,
        )
        self.ui_pipe = ui_pipe
        self.redis = Redis(host="127.0.0.1", username="test-bot", password=os.environ["REDIS_PASSWORD"], port=6379,
                           db=5)

        self.t_learner = Process(target=self.learner.run, args=tuple("Normal-astra"))
        self.t_worker = Process(target=self.worker.run, args=(self.redis, "Normal-astra"))

        if self.ui_pipe:
            self.t_pipe = Thread(target=self.handle_ui_communication)
            self.t_pipe.start()

    def check_connection(self) -> bool:
        value = 1

        self.ui_pipe.send({
            "type": "ConnTest",
            "value": value
        })

        return_ui = self.ui_pipe.recv()
        if "status" in return_ui and return_ui["status"] != "OK":
            return False

        if return_ui["value"] != value:
            return False

        return True

    def handle_ui_communication(self):
        while True:
            if self.ui_pipe.poll():
                data = self.ui_pipe.recv()

                if "type" not in data:
                    print("Invalid data received")
                    continue

                # Connection check
                if data["type"] == "RewardChange":

                    if data['weights'] is None or data['weights'] == []:
                        print(f"Rewards weights should be a list with elements, got {data['weights']})")
                        continue

                    if data['rewards'] is None or data['rewards'] == [] or len(data['rewards']) != len(data['weights']):
                        continue

                    self.rewards_weights = data["weights"]

                    saved_rewards = {}

                    with open("static/saves/rewards.json", "w") as f:
                        for key, value in zip(self.rewards, self.rewards_weights):
                            saved_rewards.setdefault(key, value)

                        f.write(json.JSONEncoder().encode(saved_rewards))

                # Start signal
                if data["type"] == "Start":
                    if not self.running:
                        self.run_both()

                # Worker start signal
                if data["type"] == "Start_w":
                    if not self.worker_running:
                        self.run_worker_only()

                # Learner start signal
                if data["type"] == "Start_l":
                    if not self.learner_running:
                        self.run_learner_only()

                # Stop signal
                if data["type"] == "Stop":
                    self.stop_both()

                # Worker stop signal
                if data["type"] == "Stop_w":
                    self.stop_worker()

                # Learner stop signal
                if data["type"] == "Stop_l":
                    self.stop_learner()

    @property
    def running(self) -> bool:
        return self.worker_running and self.learner_running

    @property
    def learner_running(self):
        return self.t_learner.is_alive()

    @property
    def worker_running(self):
        return self.t_worker.is_alive()

    def run_learner_only(self):
        try:
            print_learner("Make sure redis is running")
            with self.learner_capturer:
                self.t_learner.start()
            print_learner("Learner starting")
        except Exception as e:
            print_learner("Interruption detected, stopping learner")
            print_learner(f"The interruption is : {e}")
            self.stop_learner()

    def run_worker_only(self):
        try:
            print_worker("Make sure the learner is running")
            with self.worker_capturer:
                self.t_worker.start()
            print_worker("Worker starting")
        except Exception as e:
            print_worker("Interruption detected, stopping worker")
            print_worker(f"The interruption is : {e}")
            self.stop_worker()

    def run_both(self):

        try:
            self.run_learner_only()
            print(Fore.WHITE + "Waiting 5 seconds to start worker")

            time.sleep(5)
            self.run_worker_only()

        except Exception as e:
            print("Interruption detected, stopping both learner and worker")
            print(f"The interruption is : {e}")
            self.stop_both()

    def stop_both(self):
        self.stop_learner()
        self.stop_worker()

    def stop_learner(self):
        print("Interrupting learner")
        if self.t_learner.is_alive():
            print("Interrupted")
            self.t_learner.terminate()
            self.t_learner.join()
            print("Learner dead")
            print(self.t_learner.is_alive())

    def stop_worker(self):
        if self.t_worker.is_alive():
            self.t_worker.terminate()
            self.t_worker.join()
