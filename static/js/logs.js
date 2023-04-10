$(document).ready(function() {
    let socket = io();

    socket.on("wOutput", (data) => {
        console.log(data)
        $("#worker_log #log").text(data)
    })

    socket.on("lOutput", (data) => {
        $("#learner_log #log").text(data)
    })
})