


function add_rewards_monitoring(rewards) {
    let rewards_log = $("#rewards")
    rewards_log.empty()

    for (const [key, value] of Object.entries(rewards)) {

        let row = $("<tr class='mx-3 p-3' style='outline: black solid 1px'>")

        row.append(
            $("<td style='padding: 8px 4px; text-align: center'>").text(key)
        )
        row.append(
            $("<td style='padding: 8px 4px; text-align: center'>").text(parseFloat(value).toFixed(2))
        )

        rewards_log.append(
            row
        )
    }
}

function add_reward_configuration(rewards, rewards_weights) {
    let rewards_log = $("#rewards_config")
    for (let i = 0; i < rewards.length; i++) {

        let row = $("<tr class='mx-3 p-3' style='outline: black solid 1px'>")

        let rew = rewards[i]
        if(rewards[i].endsWith("Reward"))
            rew = rewards[i].substring(0, rewards[i].length - 6).replace(/([A-Z])/g, ' $1').trim()

        row.append(
            $("<td style='padding: 8px 4px; text-align: center'>").html(rew)
        )
        row.append(
            $("<td style='padding: 8px 4px; text-align: center'>").html("" +
                "<input value=" + parseFloat(rewards_weights[i]).toFixed(2) + " style='width: 60px; text-align: center' type='text'></input>")
        )

        rewards_log.append(
            row
        )


    }
    $("#config").append(
        $("<input type='button' value='Change weights' id='change_weight'></input>")
    )

    $("#change_weight")[0].addEventListener("click", () => {
        let new_weights = []
        $(":text").each(function () {
            new_weights.push(parseFloat(this.value))
        })

        let new_rewards = []

        $("#rewards_config td").each(function() {
            if(this.textContent !== "")
                new_rewards.push(this.textContent)
        })

        emit_to("reward_change", {
            "weights": new_weights,
            "rewards": new_rewards
        })
    })
}

function emit_to(eventName, data) {
    let socket = io();

    socket.emit(eventName, data);
}

$(document).ready(function () {
    let socket = io();


    socket.on("rewards_response", (msg) => {

        let rewards = msg.data.data.rewards;

        add_rewards_monitoring(rewards)
    })
})