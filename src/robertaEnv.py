import numpy as np
import math

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

PHI_WEIGHT = 5.0   
SURVIVAL_REWARD = 1.0      
PHI_DOT_WEIGHT = 1.00
ACTION_WEIGHT = 1.00   
KILL_REWARD = -20000       
MAX_STEPS = 2000


class RobertaEnv(gym.Env[np.ndarray, np.ndarray]):

    metadata = {
        "render_modes":["human", "rgb_array"],
        "render_fps": 120,
    }

    def __init__(self, render_mode: str | None = None):

        self._gravity = 9.8
        self._tau = 0.001
        self._throttle_converter = 12.25
        self._setpoint = 0.0
        self._equilibrium = 0.0

        self._mass_arm = 1.0
        self._length_arm = 0.4
        self._inertia_momentum = (self._mass_arm * self._length_arm ** 2)/3

        self.kinematics_integrator = ""
        self._steps = 0
        

        self._max_angle = 45.0 * math.pi / 180.0
        self._min_angle = -52.0 * math.pi / 180.0
        self._max_dangle = self._max_angle - self._min_angle

        self.x_thereshold = 2

        high = np.array(
            [
                self._max_angle,
                5 * self._max_dangle,
                self._max_angle,
            ],
            dtype=np.float32,
        )
        low = np.array(
            [
                self._min_angle,
                -5 * self._max_dangle,
                self._min_angle,
            ],
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(-1, 1, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        self.render_mode = render_mode

        # increased default size for better visualization
        self.screen_width = 1280
        self.screen_height = 900
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state: np.ndarray | None = None


    def step(self, action):
        assert self.action_space.contains(action), (
            f"{action!r} ({type(action)}) invalid"
        )
        assert self.state is not None, "Call reset before using step method."

        phi, phi_dot, setpoint = self.state

        throttle = 0.5 * float(action[0]) + 0.5
        if throttle < 0.05:
            throttle = 0.0 #Ponto de zona morta

        # store last throttle for rendering
        self._last_throttle = float(throttle)

        motor_force = throttle * self._throttle_converter
        gravity_force = self._mass_arm * self._gravity

        cosphi = np.cos(phi)
        torque = (motor_force - 0.5 * gravity_force * cosphi) * self._length_arm
        phi_acc = torque / self._inertia_momentum

        if self.kinematics_integrator == "euler":
            phi = phi + self._tau * phi_dot
            phi_dot = phi_dot + self._tau * phi_acc
        else:
            phi_dot = phi_dot + self._tau * phi_acc
            phi = phi + self._tau * phi_dot
            
        terminated = bool(
            phi < self._min_angle
            or phi > self._max_angle
            # or phi_dot > 5 * self._max_dangle
            # or phi_dot < -5 * self._max_dangle
        )

        # Keep observations within bounds even on termination.
        raw_state = np.array((phi, phi_dot, setpoint), dtype=np.float64)
        self.state = np.clip(
            raw_state, self.observation_space.low, self.observation_space.high
        )

        #punir por velocidade muito alta (esforco do motor)
        phi_dot_error = (phi_dot/(5 * self._max_dangle))**2
        #punir por distancia do objetivo
        phi_error = ((phi-setpoint)/self._max_dangle)**2
        # punicao por velocidade + punicao por distancia + punicao por sobrecarga  
        # do motor + reward por sobrevivencia
        if not terminated:
            reward = - (PHI_DOT_WEIGHT * phi_dot_error + PHI_WEIGHT * phi_error 
                        + ACTION_WEIGHT * (throttle - self._equilibrium)**2) + SURVIVAL_REWARD
        else:
            reward = np.array([KILL_REWARD])
        
        if self.render_mode == "human":
            self.render()

        self._steps += 1
        truncated = False
        if self._steps >= MAX_STEPS:
            truncated = True

        return np.array(self.state, dtype=np.float32), reward.item(), terminated, truncated, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,       
    ):
        super().reset(seed=seed)

        phi_inicial = self.np_random.uniform(
            low=self._min_angle, high=self._max_angle
        )
        # Velocidade inicial limitada a 20% (5x menor) da velocidade máxima
        phi_dot_inicial = self.np_random.uniform(
            low=-self._max_dangle / 5, high=self._max_dangle / 5
        )
        self._setpoint = self.np_random.uniform(
            low=self._min_angle, high=self._max_angle
        )

        self._equilibrium = (0.5 * self._mass_arm * self._gravity 
                            * np.cos(self._setpoint))/self._throttle_converter

        self.state = np.array(
            [
                phi_inicial, phi_dot_inicial, self._setpoint
            ],
            dtype=np.float32
        )
        # last throttle used for rendering arrows
        self._last_throttle = 0.0
        self._steps = 0

        initial_info = {
            "gravity":self._gravity,
            "tau":self._tau,
            "throttle_converter":self._throttle_converter,
            "mass_arm":self._mass_arm,
            "length_arm":self._length_arm,
            "kinematics_integrator":self.kinematics_integrator,
            "max_angle":self._max_angle,
            "min_angle":self._min_angle,
            "max_dangle":self._max_dangle,
            "phi_dot_weight": PHI_DOT_WEIGHT,
            "phi_weight": PHI_WEIGHT,
            "action_weight":ACTION_WEIGHT,
            "kill_reward": KILL_REWARD,
            "max_steps":MAX_STEPS,
        }

        if self.render_mode == "human":
            self.render()
        return self.state, initial_info
    
    def render(self):
        if self.render_mode is None:
            assert self.spec is None
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

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state is None:
            return None

        # Geometry and transforms
        phi, phi_dot, setpoint = self.state
        world_width = self.x_thereshold * 2
        scale = self.screen_width / world_width
        armlen_px = scale * self._length_arm
        arm_width_px = max(8, int(0.03 * self.screen_width))  # fixed visual width

        center_x = self.screen_width / 2
        center_y = self.screen_height / 2

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
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        arm_color = (202, 152, 101)
        gfxdraw.filled_polygon(self.surf, [c1, c2, c3, c4], arm_color)
        gfxdraw.aapolygon(self.surf, [c1, c2, c3, c4], (0, 0, 0))

        # Pivot
        gfxdraw.filled_circle(self.surf, int(center_x), int(center_y), 6, (50, 50, 50))

        # Tip point in world pixels (before screen transform)
        tip_x_world = p3_x
        tip_y_world = p3_y
        tip = to_screen(tip_x_world, tip_y_world)

        # Throttle arrow (tangential force at the tip)
        throttle = float(getattr(self, "_last_throttle", 0.0))
        ARROW_MAX = max(20, int(armlen_px * 0.6))
        th_len = ARROW_MAX * np.clip(throttle, 0.0, 1.0)
        # tangential direction (perpendicular to radial)
        dir_x, dir_y = -sinphi, cosphi
        end_th_x_world = tip_x_world + dir_x * th_len
        end_th_y_world = tip_y_world + dir_y * th_len
        end_th = to_screen(end_th_x_world, end_th_y_world)

        # line
        pygame.draw.line(self.surf, (20, 120, 255), tip, end_th, 3)
        # head triangle (smaller)
        head_size = max(6, int(0.02 * self.screen_width))
        perp_x, perp_y = -dir_y, dir_x
        head_tip = end_th
        back_x_world = end_th_x_world - dir_x * (head_size / 2)
        back_y_world = end_th_y_world - dir_y * (head_size / 2)
        left_x_world = back_x_world + perp_x * (head_size / 2)
        left_y_world = back_y_world + perp_y * (head_size / 2)
        right_x_world = back_x_world - perp_x * (head_size / 2)
        right_y_world = back_y_world - perp_y * (head_size / 2)
        head = [to_screen(left_x_world, left_y_world), head_tip, to_screen(right_x_world, right_y_world)]
        gfxdraw.filled_polygon(self.surf, head, (20, 120, 255))
        gfxdraw.aapolygon(self.surf, head, (0, 0, 0))

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
        pygame.draw.line(self.surf, (220, 30, 30), start_om, end_om, 3)
        # omega head
        head_size_o = max(6, int(0.02 * self.screen_width))
        perp_ox, perp_oy = -om_dir_y, om_dir_x
        back_om_x_world = end_om_x_world - om_dir_x * (head_size_o / 2)
        back_om_y_world = end_om_y_world - om_dir_y * (head_size_o / 2)
        left_om_x_world = back_om_x_world + perp_ox * (head_size_o / 2)
        left_om_y_world = back_om_y_world + perp_oy * (head_size_o / 2)
        right_om_x_world = back_om_x_world - perp_ox * (head_size_o / 2)
        right_om_y_world = back_om_y_world - perp_oy * (head_size_o / 2)
        head_om = [to_screen(left_om_x_world, left_om_y_world), to_screen(end_om_x_world, end_om_y_world), to_screen(right_om_x_world, right_om_y_world)]
        gfxdraw.filled_polygon(self.surf, head_om, (220, 30, 30))
        gfxdraw.aapolygon(self.surf, head_om, (0, 0, 0))

        # Blit and present
        self.screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes =(1, 0, 2)
            )
        
    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
    