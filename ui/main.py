import copy
import re
from abc import ABC, abstractmethod
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Union, Any, List

import numpy as np
import pygame
import rlgym.utils.common_values as constants
from pygame import Surface
from pygame.font import Font
from pygame.sysfont import SysFont
from pygame.time import Clock
from rlgym.envs import Match
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 128, 128)
BLUE = (0, 170, 255)
DARK_BLUE = (0, 153, 255)

ORANGE = (255, 214, 153)
DARK_ORANGE = (255, 153, 0)


class Binding:
    def __init__(self, attr_name: str, func = None):
        self.linked_attr = attr_name
        self.func = func

    def act(self, args):
        if self.func:
            return str(self.func(args))
        else:
            return str(args)


class MultiBinding:
    def __init__(self, bindings: List[Binding], stringformat: str):
        self.bindings = bindings
        self.strFormat = stringformat

    def act(self, args: List):
        matches = re.findall(r'({[^{]*?)(?=})}', self.strFormat, re.I | re.MULTILINE)

        final_string = self.strFormat

        for i in range(len(self.bindings)):
            binding = self.bindings[i]
            final_string = final_string.replace(matches[i] + "}", binding.act(args[i]))
        return final_string


class Widget(ABC):
    def __init__(self, location, dimensions, data_context=None):
        self.location = location
        self.dimensions = dimensions
        self.data_context = data_context

    @abstractmethod
    def draw(self, surface: Surface):
        pass

    def resize(self, old_dim, event):
        x_ratio = event.w / old_dim[0]
        y_ratio = event.h / old_dim[1]

        self.location[0] *= x_ratio
        self.location[1] *= y_ratio

        self.dimensions[0] *= x_ratio
        self.dimensions[1] *= y_ratio


class FieldViewer(Widget):
    POS_X_STD = 8192
    POS_Y_STD = 12000

    def __init__(self, child_conn: Connection, location, dimensions, data_context=None):
        super().__init__(location, dimensions, data_context)
        self.conn = child_conn
        self.player_size = 5

        self.mock_players = None
        self.players = np.array([])
        self.ball = None

    def field_coord_to_screen(self, coord):

        posX = self.location[0] + (coord.item(0) + self.POS_X_STD * 0.5) / self.POS_X_STD * self.dimensions[0]
        posY = self.location[1] + (coord.item(1) + self.POS_Y_STD * 0.5) / self.POS_Y_STD * self.dimensions[1]

        return np.array([posX, posY, coord.item(2)])

    def draw(self, surface: Surface):
        # if last_len != len(data) - 1:
        #     last_len = len(data)
        #     trails = []
        #     for i in range(last_len):
        #         trails.append([])

        surface.fill(WHITE)
        bg = pygame.image.load("ui/bg_field.png").convert_alpha()
        bg = pygame.transform.scale(bg, self.dimensions).convert_alpha(surface)
        surface.blit(bg, self.location)

        # players = data[0:-1]

        players = self.players

        for player in players:
            position = self.field_coord_to_screen(player.car_data.position)

            spectated = False

            posX = position.item(0)
            posY = position.item(1)

            color = ORANGE
            dark_color = DARK_ORANGE
            if player.team_num == constants.BLUE_TEAM:
                color = BLUE
                dark_color = DARK_BLUE

            if player.car_id == 1:
                spectated = True

            radius = self.player_size

            forward = player.car_data.forward()
            forward = pygame.math.Vector2(forward[0], forward[1]).normalize()

            vForward = forward
            vCenter = pygame.math.Vector2(posX, posY)
            angle = pygame.math.Vector2().angle_to(vForward)

            points = [(0.2 / 10 * radius, -0.916 / 10 * radius),
                      (0.2 / 10 * radius, 0.916 / 10 * radius),
                      (3.7 / 10 * radius, 0.0 / 10 * radius)]

            outline_points = [(0.2 / 10 * radius, -.976 / 10 * radius),
                              (0.2 / 10 * radius, .976 / 10 * radius),
                              (4.2 / 10 * radius, 0.0 / 10 * radius)]

            rotated_point = [pygame.math.Vector2(p).rotate(angle) for p in points]
            rotated_outline_point = [pygame.math.Vector2(p).rotate(angle) for p in outline_points]
            scale = 10
            triangle_points = [(vCenter + p * scale) for p in rotated_point]
            outline_triangle_points = [(vCenter + p * scale) for p in rotated_outline_point]

            if spectated:
                pygame.draw.circle(pygame.display.get_surface(), color, (posX, posY), radius=radius + 2)
            pygame.draw.circle(pygame.display.get_surface(), dark_color, (posX, posY), radius=radius)

            if spectated:
                pygame.draw.polygon(surface, color, outline_triangle_points)
            pygame.draw.polygon(surface, dark_color, triangle_points)

        if self.ball:
            ball_pos = self.field_coord_to_screen(self.ball.position)

            pygame.draw.circle(surface, WHITE, (ball_pos.item(0), ball_pos.item(1)), radius=20)
            font = pygame.font.SysFont("Calibri", 15)
            text = font.render(f"B", True, BLACK)
            text_rect = text.get_rect()
            text_rect.center = (ball_pos.item(0), ball_pos.item(1))
            surface.blit(text, text_rect)


class TextBox(Widget):
    def __init__(self, text: Union[str, Binding, MultiBinding], location, dimensions, font: Union[Font, SysFont],
                 data_context=None, show_invisible_cars: bool = False):
        super().__init__(location, dimensions, data_context)
        self.show_invisible_cars = show_invisible_cars
        self.text = text
        self.font = font

    def draw(self, surface: Surface):
        if isinstance(self.text, Binding):
            if not self.data_context:
                raise TypeError("Expected Object, got " + str(type(self.data_context).__name__))

            if not hasattr(self.data_context, self.text.linked_attr):
                raise ValueError(
                    f"The type {type(self.data_context)} doesn't contains a {self.text.linked_attr} attribute")

            if isinstance(self.data_context, type(PlayerData())):
                if not self.show_invisible_cars and self.data_context.car_id == -1:
                    return

            text = self.font.render(str(self.text.act(getattr(self.data_context, self.text.linked_attr))), True, BLACK)

        elif isinstance(self.text, MultiBinding):
            if not self.data_context:
                raise TypeError("Expected Object, got " + str(type(self.data_context).__name__))

            text = self.font.render(str(self.text.act(
                [getattr(self.data_context, i.linked_attr) if hasattr(self.data_context, i.linked_attr) else '' for i in
                 self.text.bindings])), True, BLACK)
        else:
            text = self.font.render(self.text, True, BLACK)

        text_rect = text.get_rect()
        text_rect.topleft = self.location
        text_rect.width, text_rect.height = self.dimensions
        surface.blit(text, text_rect)


class Software(Process):

    def __init__(self):
        super().__init__()

        self.running = True
        self.child_conn = None

        # Width/Height
        self.dimensions = [1000, 500]

    def run(self) -> None:
        self.run_ui()

    def run_ui(self):

        widgets = []

        pygame.init()

        surface = pygame.display.set_mode(self.dimensions, flags=pygame.RESIZABLE)
        pygame.display.set_caption("Rocket-learn window")
        surface.fill(WHITE)
        pygame.display.flip()

        clock = Clock()

        bg = pygame.image.load("ui/bg_field.png").convert_alpha()
        bg = pygame.transform.scale(bg, self.dimensions).convert_alpha(surface)
        surface.blit(bg, (0, 0))

        p1 = PlayerData()
        p1.car_id = 1
        p1.car_data = PhysicsObject(position=np.array([500, 1020, 60]),
                                    quaternion=np.array([0, 1, 1, 0]))

        p2 = PlayerData()
        p2.car_id = 2
        p2.car_data = PhysicsObject(position=np.array([-500, -1020, 60]),
                                    quaternion=np.array([0, 0, 1, 0.0000463]),
                                    linear_velocity=np.array([45, 45, 0]))

        p3 = PlayerData()
        p3.car_id = 3
        p3.team_num = constants.BLUE_TEAM
        p3.car_data = PhysicsObject(position=np.array([500, -1020, 60]),
                                    quaternion=np.array([1, 0, 0, 0]))

        players = [
            p1, p2, p3
        ]

        empty_player = PlayerData()

        trails = []

        #
        # print(p1.car_data.forward())
        # print(p2.car_data.forward())
        # print(p3.car_data.forward())

        width = surface.get_height() * 5 / 6 - 20

        f = FieldViewer(self.child_conn, [surface.get_width() - width, 10], [width, surface.get_height() - 20])

        text_boost_title = TextBox(
            text="Boost amounts :",
            location=[5, 10],
            dimensions=[20, 100],
            font=SysFont("Calibri", 15)
        )

        text_boosts = [
            TextBox(MultiBinding(
                [Binding("car_id"), Binding("boost_amount", func=lambda v: int(v * 100))], "Car {0} : {1}"
            ), [10, 30], [20, 100], SysFont("Calibri", 15), data_context=empty_player),
            TextBox(MultiBinding(
                [Binding("car_id"), Binding("boost_amount", func=lambda v: int(v * 100))], "Car {1} : {0}"
            ), [10, 50], [20, 100], SysFont("Calibri", 15), data_context=empty_player),
            TextBox(MultiBinding(
                [Binding("car_id"), Binding("boost_amount", func=lambda v: int(v * 100))], "Car {1} : {0}"
            ), [10, 70], [20, 100], SysFont("Calibri", 15), data_context=empty_player),
            TextBox(MultiBinding(
                [Binding("car_id"), Binding("boost_amount", func=lambda v: int(v * 100))], "Car {1} : {0}"
            ), [10, 90], [20, 100], SysFont("Calibri", 15), data_context=empty_player),
            TextBox(MultiBinding(
                [Binding("car_id"), Binding("boost_amount", func=lambda v: int(v * 100))], "Car {1} : {0}"
            ), [10, 110], [20, 100], SysFont("Calibri", 15), data_context=empty_player),
            TextBox(MultiBinding(
                [Binding("car_id"), Binding("boost_amount", func=lambda v: int(v * 100))], "Car {1} : {0}"
            ), [10, 130], [20, 100], SysFont("Calibri", 15), data_context=empty_player),
        ]

        widgets.append(f)
        widgets.append(text_boost_title)
        widgets.extend(text_boosts)

        # f.mock_players = players

        last_len = 0

        while self.running:

            for widget in widgets:
                if isinstance(widget.data_context, type(PlayerData())):
                    if widget.data_context.car_id != -1:
                        widget.draw(surface)
                    else:
                        continue

                widget.draw(surface)

            if self.child_conn and self.child_conn.poll():
                data = self.child_conn.recv()

                ball = data[-1]
                f.ball = ball

                if last_len != len(data[0:-1]):
                    last_len = len(data[0:-1])
                    for text_boost in text_boosts:
                        text_boost.data_context = empty_player

                f.players = data[0:-1]

                for index, player in enumerate(data[0:-1]):
                    text_boosts[index].data_context = player

            if len(pygame.event.get(pygame.QUIT)) > 0:
                pygame.quit()
                self.running = False

            resize_events = pygame.event.get(pygame.VIDEORESIZE)
            if len(resize_events) > 0:
                event = resize_events[0]
                # There's some code to add back window content here.
                surface = pygame.display.set_mode((event.w, event.h),
                                                  pygame.RESIZABLE)
                old_dimensions = copy.copy(self.dimensions)
                self.dimensions = (event.w, event.h)
                # f.dimensions = (event.h * 5 / 6, event.h - 20)
                # f.location = (event.w - event.h * 5 / 6, 10)

                for widget in widgets:
                    widget.resize(old_dimensions, event)

                surface.fill(WHITE)

            clock.tick(60)
            pygame.display.flip()


class AstraMatch(Match):
    def __init__(self,
                 reward_function,
                 terminal_conditions,
                 obs_builder,
                 action_parser,
                 state_setter,
                 software,
                 team_size=1,
                 tick_skip=8,
                 game_speed=100,
                 gravity=1,
                 boost_consumption=1,
                 spawn_opponents=False,
                 ):
        super().__init__(reward_function, terminal_conditions, obs_builder, action_parser, state_setter, team_size,
                         tick_skip, game_speed, gravity, boost_consumption, spawn_opponents)
        self.software = software
        self.parent_conn, child_conn = Pipe()
        software.child_conn = child_conn
        self.package_wait = 8
        self.software.start()

    def build_observations(self, state: GameState) -> Union[Any, List]:
        obs = super().build_observations(state)

        if self.package_wait != 0:
            self.package_wait -= 1
            return obs

        data = np.concatenate(([player for player in state.players], [state.ball]))

        self.parent_conn.send(data)
        self.package_wait = 8

        return obs

#
#
# s = Software()
# s.run_ui()
