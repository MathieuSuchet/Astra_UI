let send_start_signal = () => {
    let socket = io();

    socket.emit("start", {
        "type": "Start"
    })
}

let send_start_signal_learner = () => {
    let socket = io();

    socket.emit("start", {
        "type": "Start_l"
    })
}

let send_start_signal_worker = () => {
    let socket = io();

    socket.emit("start", {
        "type": "Start_w"
    })
}

let send_stop_signal = () => {
    let socket = io();

    socket.emit("stop", {
        "type": "Stop"
    })
}

let send_stop_signal_learner = () => {
    let socket = io();

    socket.emit("stop", {
        "type": "Stop_l"
    })
}

let send_stop_signal_worker = () => {
    let socket = io();

    socket.emit("stop", {
        "type": "Stop_w"
    })
}

$(document).ready(function() {
    let socket = io();

    socket.on("started", (msg) => {
        if(msg.state){
            $("#astra_both").html($("#stop_both_btn").html())
        }
        else{
            //Fail
        }
    })

    socket.on("started_l", (msg) => {
        if(msg.state){
            $("#astra_learner").html($("#stop_learner_btn").html())
        }
        else{
            //Fail
        }
    })

    socket.on("started_w", (msg) => {
        if(msg.state){
            $("#astra_worker").html($("#stop_worker_btn").html())
        }
        else{
            //Fail
        }
    })
})