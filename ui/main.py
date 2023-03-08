from multiprocessing import Pipe, Process
from typing import Union, Any, List

import numpy as np
import pygame
import rlgym.utils.common_values as constants
from pygame.time import Clock
from rlgym.envs import Match
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject


class Software(Process):
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 128, 128)
    BLUE = (0, 170, 255)
    DARK_BLUE = (0, 153, 255)

    ORANGE = (255, 214, 153)
    DARK_ORANGE = (255, 153, 0)

    POS_X_STD = 8192
    POS_Y_STD = 12000

    def __init__(self):
        super().__init__()

        self.running = True
        self.child_conn = None

        width = 580
        # Width/Height
        self.dimensions = (width, 80 * width / 58)

    def run(self) -> None:
        self.run_ui()

    def field_coord_to_screen(self, coord):

        posX = (coord.item(0) + self.POS_X_STD * 0.5) / self.POS_X_STD * self.dimensions[0]
        posY = (coord.item(1) + self.POS_Y_STD * 0.5) / self.POS_Y_STD * self.dimensions[1]

        return np.array([posX, posY, coord.item(2)])

    def run_ui(self):

        pygame.init()

        surface = pygame.display.set_mode(self.dimensions)
        pygame.display.set_caption("Rocket-learn window")
        surface.fill(self.WHITE)
        pygame.display.flip()

        clock = Clock()

        bg = pygame.image.load("ui/bg_field.png").convert_alpha()
        bg = pygame.transform.scale(bg, self.dimensions).convert_alpha(surface)
        surface.blit(bg, (0, 0))

        trails = []

        p1 = PlayerData()
        p1.car_id = 1
        p1.car_data = PhysicsObject(position=np.array([500, 1020, 60]),
                                    quaternion=np.array([0, 1, 1, 0]))

        p2 = PlayerData()
        p2.car_id = 2
        p2.car_data = PhysicsObject(position=np.array([-500, -1020, 60]),
                                    quaternion=np.array([0, 0, 1, 0.0000463]))

        p3 = PlayerData()
        p3.car_id = 3
        p3.team_num = constants.BLUE_TEAM
        p3.car_data = PhysicsObject(position=np.array([500, -1020, 60]),
                                    quaternion=np.array([1, 0, 0, 0]))

        last_len = 0

        players = [
            p1, p2, p3
        ]
        #
        # print(p1.car_data.forward())
        # print(p2.car_data.forward())
        # print(p3.car_data.forward())

        while self.running:

            if self.child_conn and self.child_conn.poll():
                data = self.child_conn.recv()

                if last_len != len(data) - 1:
                    last_len = len(data)
                    trails = []
                    for i in range(last_len):
                        trails.append([])

                surface.fill(self.WHITE)
                bg = pygame.image.load("ui/bg_field.png").convert_alpha()
                bg = pygame.transform.scale(bg, self.dimensions).convert_alpha(surface)
                surface.blit(bg, (0, 0))

                players = data[0:-1]
                ball = data[-1]

                for player in players:
                    position = self.field_coord_to_screen(player.car_data.position)

                    spectated = False

                    posX = position.item(0)
                    posY = position.item(1)

                    color = self.ORANGE
                    dark_color = self.DARK_ORANGE
                    if player.team_num == constants.BLUE_TEAM:
                        color = self.BLUE
                        dark_color = self.DARK_BLUE

                    if player.car_id == 1:
                        spectated = True

                    radius = 10

                    forward = player.car_data.forward()
                    forward = pygame.math.Vector2(forward[0], forward[1]).normalize()

                    vForward = forward
                    vCenter = pygame.math.Vector2(posX, posY)
                    angle = pygame.math.Vector2().angle_to(vForward)

                    points = [(0.2, -0.866), (0.2, 0.866), (3.7, 0.0)]
                    outline_points = [(0.2, -1.166), (0.2, 1.166), (4.5, 0.0)]
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

                ball_pos = self.field_coord_to_screen(ball.position)

                pygame.draw.circle(surface, self.WHITE, (ball_pos.item(0), ball_pos.item(1)), radius=20)
                font = pygame.font.SysFont("Calibri", 15)
                text = font.render(f"B", True, self.BLACK)
                text_rect = text.get_rect()
                text_rect.center = (ball_pos.item(0), ball_pos.item(1))
                surface.blit(text, text_rect)

            if len(pygame.event.get(pygame.QUIT)) > 0:
                pygame.quit()
                self.running = False

            clock.tick(60)
            pygame.display.flip()


class AstraMatch(Match):

    def __init__(self, reward_function, terminal_conditions, obs_builder, action_parser, state_setter, software,
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
