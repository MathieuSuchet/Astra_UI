import argparse
import json
from multiprocessing import Lock, Pipe
from threading import Thread
from typing import Any

import numpy
from colorama import Fore
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward, EventReward, TouchBallReward, \
    FaceBallReward
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition

from CustomStateSetter import ProbabilisticStateSetter
from Rewards.CustomRewards import AerialTouchReward, PressureReward, GoalScoreSpeed, KickoffReward, SaveBoostReward, \
    PosessionReward
from astra import Astra
from obs.AstraObs import AstraObs

astra_pipe, ui_pipe = Pipe(duplex=True)


class ExpandAdvancedObs(AstraObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)


rewards = (
    AerialTouchReward(),
    PressureReward(),
    GoalScoreSpeed(),
    KickoffReward(),
    SaveBoostReward(),
    PosessionReward(),
    EventReward(
        team_goal=100,
        concede=-100,
        demo=0.25,
        boost_pickup=1.5,
        save=3
    )
)

rewards_weights = (0.15, 0.2, 0.042, 1.0, 0.012, 1.3, 0.45)

try:
    with open("static/saves/rewards.json", "r") as f:
        data = json.loads(f.read())
        rewards_weights = data.values()
except IOError:
    print("No rewards saved")
except json.decoder.JSONDecodeError:
    print("Cannot decode rewards, ignoring..")

a = Astra(
    obs_builder=ExpandAdvancedObs(),
    action_parser=DiscreteAction(),
    state_setter=ProbabilisticStateSetter(),
    rewards=rewards,
    rewards_weights=rewards_weights,
    terminal_conditions=[GoalScoredCondition(), TimeoutCondition(2000)],
    ui_pipe=ui_pipe
)

async_mode = None

app = Flask(__name__)
app.secret_key = b"The_app_secret_key"
socketio = SocketIO(app, async_mode=async_mode, logger=True, engineio_logger=True)

thread = None
thread_lock = Lock()


@app.route('/')
def index():
    return render_template("index.html", bot=a, async_mode=socketio.async_mode)


@app.route("/agents_summaries", methods=["GET", "POST"])
def agents_summaries():
    if request.headers.get("Content-Type") == "application/json":
        print(f"Got json : {request.json}")
        socketio.emit("my_response", {'data': json.loads(request.json)["data"]})
    return render_template("agent_summaries.html", bot=a)


@app.route("/bot_connection")
def bot_connection():
    print(Fore.WHITE + f"Got new state : {json.loads(request.json)}")
    a.is_alive = json.loads(request.json)
    socketio.emit("bot_connection", {"state": json.loads(request.json)}, namespace="/about")
    return {
        "returnCode": 200,
        "message": "Bot state received"
    }


def handle_astra_communication():
    while True:
        socketio.sleep(3)
        if astra_pipe.poll():
            data = astra_pipe.recv()

            if "type" not in data:
                print(Fore.RED + "Invalid data received")
                continue

            # Connection check
            if data["type"] == "ConnTest":
                astra_pipe.send({
                    "type": "ConnTest",
                    "status": "OK",
                    "value": data["value"]
                })

            # Log change
            if data["type"] == "lOutput":
                socketio.emit("lOutput", new_log=data["data"])

            if data["type"] == "wOutput":
                socketio.emit("wOutput", new_log=data["data"])


@app.route("/logs", methods=["GET", "POST"])
def logs():
    print(a.learner_capturer)
    print(a.worker_capturer)
    return render_template("logs.html", bot=a)


@socketio.on("reward_change")
def change(data):
    astra_pipe.send({
        "type": "RewardChange",
        "weights": data["weights"],
        "rewards": data["rewards"]
    })


@app.route("/rewards", methods=["GET", "POST"])
def rewards():
    r = [type(rew).__name__ for rew in a.rewards]
    rw = [w for w in a.rewards_weights]
    if request.headers.get("Content-Type") == "application/json":
        if "is_final" in json.loads(request.json)["data"]:
            # a.rewards = json.loads(request.json)
            socketio.emit("rewards_response", {'data': json.loads(request.json)})

    # if a.rewards != {}:
    #     return render_template(template_name_or_list="rewards.html", bot=a,
    #                            rewards=a.rewards["data"]["rewards"])
    return render_template(template_name_or_list="rewards.html", bot=a, rewards_config=r, rewards_weights=rw)


@socketio.on("start")
def start(data):
    print(Fore.WHITE + "Start signal detected, sending to astra")
    astra_pipe.send(data)


@socketio.on("stop")
def stop(data):
    print(Fore.WHITE + "Stop signal detected, sending to astra")
    astra_pipe.send(data)


@app.route("/about")
def about():
    return render_template("about.html", bot=a)


@socketio.event
def connect():
    global thread
    with thread_lock:
        if thread is None:
            print("Starting thread")
            thread = socketio.start_background_task(target=handle_astra_communication)


parser = argparse.ArgumentParser(
    prog="Astra",
    description="Start the bot and the UI (doesn't run the learner/worker)",
    add_help=True
)

parser.add_argument("--start-ui", action="store_true")
parser.add_argument("--start-astra", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()

    t_astra = Thread(target=a.run_both)

    try:
        if args.start_astra:
            print("Astra starting")
            t_astra.start()

        if args.start_ui:
            print("UI starting")
            socketio.run(app, debug=True, host="127.0.0.1", port=5000)

    except KeyboardInterrupt:
        print("Interrupting")
        socketio.stop()

        if t_astra.is_alive():
            t_astra.join(5)
