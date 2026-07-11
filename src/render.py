import math

import numpy as np
from gymnasium import logger
from gymnasium.error import DependencyNotInstalled

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robertaEnv import RobertaEnv

X_THERESHOLD = 2

def render(Env: 'RobertaEnv') -> None:
    if Env.render_mode is None:
            assert Env.spec is None
            logger.warn(
                "You are calling render() without specifying a render_mode."
                " Pass `render_mode='rgb_array'` or `render_mode='human'` when creating the env."
            )
            return

    try:
        import pygame
        from pygame import gfxdraw
    except ImportError as e:
        raise DependencyNotInstalled (
            'pygame is not installed, run `pip install :"gymnasium[classic-control]"`'
        ) from e

    if Env.screen is None:
        pygame.init()
        if Env.render_mode == "human":
            pygame.display.init()
            Env.screen = pygame.display.set_mode(
                (Env.screen_width, Env.screen_height)
            )
        else:
            Env.screen = pygame.Surface((Env.screen_width, Env.screen_height))

    if Env.clock is None:
        Env.clock = pygame.time.Clock()

    if Env.state is None:
        return None

    # Geometry and transforms
    phi, phi_dot, setpoint, _ = Env.state
    world_width = Env.x_thereshold * 2
    scale = Env.screen_width / world_width
    armlen_px = scale * Env._length_arm
    arm_width_px = max(8, int(0.03 * Env.screen_width))  # fixed visual width

    center_x = Env.screen_width / 2
    center_y = Env.screen_height / 2

    cosphi = math.cos(phi)
    sinphi = math.sin(phi)

    # Rectangle corners in local (arm) coordinates
    x1, y1 = 0, -arm_width_px / 2
    x2, y2 = 0, arm_width_px / 2
    x3, y3 = armlen_px, arm_width_px / 2
    x4, y4 = armlen_px, -arm_width_px / 2

    def rot(x, y):
        rx = x * cosphi - y * sinphi
        ry = x * sinphi + y * cosphi
        return rx, ry

    p1_x, p1_y = rot(x1, y1)
    p2_x, p2_y = rot(x2, y2)
    p3_x, p3_y = rot(x3, y3)
    p4_x, p4_y = rot(x4, y4)

    def to_screen(px, py):
        return (int(center_x + px), int(center_y - py))

    c1 = to_screen(p1_x, p1_y)
    c2 = to_screen(p2_x, p2_y)
    c3 = to_screen(p3_x, p3_y)
    c4 = to_screen(p4_x, p4_y)

    # Draw to an offscreen surface then blit
    Env.surf = pygame.Surface((Env.screen_width, Env.screen_height))
    Env.surf.fill((255, 255, 255))

    arm_color = (202, 152, 101)
    gfxdraw.filled_polygon(Env.surf, [c1, c2, c3, c4], arm_color)
    gfxdraw.aapolygon(Env.surf, [c1, c2, c3, c4], (0, 0, 0))

    # Pivot
    gfxdraw.filled_circle(Env.surf, int(center_x), int(center_y), 6, (50, 50, 50))

    # Tip point in world pixels (before screen transform)
    tip_x_world = p3_x
    tip_y_world = p3_y
    tip = to_screen(tip_x_world, tip_y_world)

    # Throttle arrow (tangential force at the tip)
    throttle = float(getattr(Env, "_last_throttle", 0.0))
    ARROW_MAX = max(20, int(armlen_px * 0.6))
    th_len = ARROW_MAX * np.clip(throttle, 0.0, 1.0)
    # tangential direction (perpendicular to radial)
    dir_x, dir_y = -sinphi, cosphi
    end_th_x_world = tip_x_world + dir_x * th_len
    end_th_y_world = tip_y_world + dir_y * th_len
    end_th = to_screen(end_th_x_world, end_th_y_world)

    # line
    pygame.draw.line(Env.surf, (20, 120, 255), tip, end_th, 3)
    # head triangle (smaller)
    head_size = max(6, int(0.02 * Env.screen_width))
    perp_x, perp_y = -dir_y, dir_x
    head_tip = end_th
    back_x_world = end_th_x_world - dir_x * (head_size / 2)
    back_y_world = end_th_y_world - dir_y * (head_size / 2)
    left_x_world = back_x_world + perp_x * (head_size / 2)
    left_y_world = back_y_world + perp_y * (head_size / 2)
    right_x_world = back_x_world - perp_x * (head_size / 2)
    right_y_world = back_y_world - perp_y * (head_size / 2)
    head = [to_screen(left_x_world, left_y_world), head_tip, to_screen(right_x_world, right_y_world)]
    gfxdraw.filled_polygon(Env.surf, head, (20, 120, 255))
    gfxdraw.aapolygon(Env.surf, head, (0, 0, 0))

    # Angular velocity arrow (tangential)
    omega = float(phi_dot)
    tang_dir_x, tang_dir_y = -sinphi, cosphi
    omega_scale = armlen_px * 0.6
    om_len = min(omega_scale, abs(omega) * (armlen_px * 0.5))
    sign = np.sign(omega) if omega != 0 else 1.0
    om_dir_x, om_dir_y = tang_dir_x * sign, tang_dir_y * sign
    start_om_x_world = tip_x_world - om_dir_x * 8
    start_om_y_world = tip_y_world - om_dir_y * 8
    end_om_x_world = start_om_x_world + om_dir_x * om_len
    end_om_y_world = start_om_y_world + om_dir_y * om_len
    start_om = to_screen(start_om_x_world, start_om_y_world)
    end_om = to_screen(end_om_x_world, end_om_y_world)
    pygame.draw.line(Env.surf, (220, 30, 30), start_om, end_om, 3)
    # omega head
    head_size_o = max(6, int(0.02 * Env.screen_width))
    perp_ox, perp_oy = -om_dir_y, om_dir_x
    back_om_x_world = end_om_x_world - om_dir_x * (head_size_o / 2)
    back_om_y_world = end_om_y_world - om_dir_y * (head_size_o / 2)
    left_om_x_world = back_om_x_world + perp_ox * (head_size_o / 2)
    left_om_y_world = back_om_y_world + perp_oy * (head_size_o / 2)
    right_om_x_world = back_om_x_world - perp_ox * (head_size_o / 2)
    right_om_y_world = back_om_y_world - perp_oy * (head_size_o / 2)
    head_om = [to_screen(left_om_x_world, left_om_y_world), to_screen(end_om_x_world, end_om_y_world), to_screen(right_om_x_world, right_om_y_world)]
    gfxdraw.filled_polygon(Env.surf, head_om, (220, 30, 30))
    gfxdraw.aapolygon(Env.surf, head_om, (0, 0, 0))

    # Blit and present
    Env.screen.blit(Env.surf, (0, 0))

    if Env.render_mode == "human":
        pygame.event.pump()
        Env.clock.tick(Env.metadata["render_fps"])
        pygame.display.flip()
    elif Env.render_mode == "rgb_array":
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(Env.screen)), axes =(1, 0, 2)
        )