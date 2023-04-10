let loadedButton = () => {
    let c = $("canvas")[0]
    let context = c.getContext("2d")

    let image = $("#bg_field")[0]
    context.drawImage(image, 0, 0)
}

$(document).ready(function () {
    let socket = io();

    $("#bg_field").onload = loadedButton;

    function roundList(list, rounding) {
        list[0] = list[0].toFixed(rounding)
        list[1] = list[1].toFixed(rounding)
        list[2] = list[2].toFixed(rounding)
        return list
    }

    socket.on("my_response", function (msg) {

        let orangeLog = $("#orange_log")
        let waiting = $("#waiting-data")[0]
        waiting.style.visibility = "collapse"
        orangeLog.empty()

        let orange_p = msg.data.players.filter(x => x.team_num === 1)

        orange_p.forEach(elt => {
            let position = roundList(elt.car_data.position, 2)
            let lin_speed = roundList(elt.car_data.linear_velocity, 2)

            let color = "#ff8822"
            let font_color = "#ffffff"

            let row = $("<tr style='outline: 1px solid black;'>")
            row[0].style.background = color
            row[0].style.color = font_color

            row.html("<td style='width: 70px'>" + elt.car_id + "</td>" +
                "<td style='width: 84px'>" + position[0] + "</td>" +
                "<td style='width: 84px'>" + position[1] + "</td>" +
                "<td style='width: 84px'>" + position[2] + "</td>" +
                "<td style='width: 20px'/>" +
                "<td style='width: 84px'>" + lin_speed[0] + "</td>" +
                "<td style='width: 84px'>" + lin_speed[1] + "</td>" +
                "<td style='width: 84px'>" + lin_speed[2] + "</td>" +
                "<td style='width: 20px'/>" +
                "<td style='width: 60px'>" + (parseFloat(elt.boost_amount) * 100).toFixed(0) + "</td>")
            orangeLog.append(row)

        })

        let blueLog = $("#blue_log")
        blueLog.empty()

        let blue_p = msg.data.players.filter(x => x.team_num === 0)

        blue_p.forEach(elt => {
            let position = roundList(elt.car_data.position, 2)
            let lin_speed = roundList(elt.car_data.linear_velocity, 2)

            let color = "#2222ff"
            let font_color = "#ffffff"

            let row = $("<tr style='color: black;outline: 1px solid black;'>")
            row[0].style.background = color
            row[0].style.color = font_color

            row.html("<td style='width: 70px;'>" + elt.car_id + "</td>" +
                "<td style='width: 84px'>" + position[0] + "</td>" +
                "<td style='width: 84px'>" + position[1] + "</td>" +
                "<td style='width: 84px'>" + position[2] + "</td>" +
                "<td style='width: 20px'/>" +
                "<td style='width: 84px'>" + lin_speed[0] + "</td>" +
                "<td style='width: 84px'>" + lin_speed[1] + "</td>" +
                "<td style='width: 84px'>" + lin_speed[2] + "</td>" +
                "<td style='width: 20px'/>" +
                "<td style='width: 60px'>" + (parseFloat(elt.boost_amount) * 100).toFixed(0) + "</td>")
            blueLog.append(row)
        })


        const POS_X_STD = 8190
        const POS_Y_STD = 12000

        function field_to_screen(context, coords) {
            let c = [parseFloat(coords[0]), parseFloat(coords[1])]
            let posX = (c[0] + POS_X_STD * 0.5) / POS_X_STD * context.canvas.width
            let posY = (c[1] + POS_Y_STD * 0.5) / POS_Y_STD * context.canvas.height

            return [posX, posY]
        }

        function draw_player(context, element, image) {
            let screenPos = field_to_screen(context, element.car_data.position)
            let forward = element.car_data.forward

            context.save()

            let angle = Math.atan2(forward[1], forward[0])

            let width = 20
            let height = 40

            context.translate(screenPos[0] + width / 2, screenPos[1] + height / 2)
            context.rotate(angle)
            context.translate(-screenPos[0] - width / 2, -screenPos[1] - height / 2)
            console.log("Drawing image at " + screenPos)
            context.drawImage(image, screenPos[0], screenPos[1], width, height)

            context.restore()
        }

        let players = msg.data.players
        //Update canvas
        let canvas = $("canvas")[0]
        let ctx = canvas.getContext("2d")

        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
        ctx.fillStyle = '#ffffff'
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height)

        let bg = $("#bg_field")[0]
        ctx.drawImage(bg, 0, 0)
        console.log("Drew field")

        let blue_octane_img = $("#blue_car")[0]
        let orange_octane_img = $("#orange_car")[0]
        let white_octane_img = $("#car")[0]

        let blue_players = players.filter((x) => x.team_num === 0)
        console.log(blue_players)

        blue_octane_img.onload = () => {
            blue_players.forEach(element => {
                console.log("Drew blue")
                draw_player(ctx, element, blue_octane_img)
            })

        }
        blue_players.forEach(element => {
            console.log("K and here ?")
            draw_player(ctx, element, blue_octane_img)
        })

        let orange_players = players.filter((x) => x.team_num === 1)

        orange_octane_img.onload = () => {
            orange_players.forEach(element => {
                draw_player(ctx, element, orange_octane_img)
            })
        }
        orange_players.forEach(element => {
            draw_player(ctx, element, orange_octane_img)
        })

        let other_players = players.filter((x) => x.team_num !== 0 && x.team_num !== 1)
        white_octane_img.onload = () => {
            other_players.forEach(element => {
                draw_player(ctx, element, white_octane_img)
            })
        };
        other_players.forEach(element => {
            draw_player(ctx, element, white_octane_img)
        })


    })

})
