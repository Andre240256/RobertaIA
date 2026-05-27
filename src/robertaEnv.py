import numpy as np
import math

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

PHI_DOT_WEIGHT = 0.1
PHI_WEIGHT = 2.5
ACTION_WEIGHT = 0.1
KILL_REWARD = -2000
MAX_STEPS = 1000


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
            ],
            dtype=np.float32,
        )
        low = np.array(
            [
                self._min_angle,
                -5 * self._max_dangle,
            ],
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(-1, 1, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state: np.ndarray | None = None


    def step(self, action):
        assert self.action_space.contains(action), (
            f"{action!r} ({type(action)}) invalid"
        )
        assert self.state is not None, "Call reset before using step method."

        phi, phi_dot = self.state

        throttle = 0.5 * float(action[0]) + 0.5
        if throttle < 0.05:
            throttle = 0.0 #Ponto de zona morta

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
        raw_state = np.array((phi, phi_dot), dtype=np.float64)
        self.state = np.clip(
            raw_state, self.observation_space.low, self.observation_space.high
        )


        phi_dot_error = (phi_dot/(5 * self._max_dangle))**2
        phi_error = ((phi-self._setpoint)/self._max_dangle)**2

        if not terminated:
            reward = - (PHI_DOT_WEIGHT * phi_dot_error + PHI_WEIGHT * phi_error + ACTION_WEIGHT * (throttle - self._equilibrium)**2)
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
                phi_inicial, phi_dot_inicial
            ],
            dtype=np.float32
        )
        self._steps = 0

        if self.render_mode == "human":
            self.render()
        return self.state, {"setpoint":self._setpoint}
    
    def render(self):
        if self.render_mode is None:
            assert self.spec is None
            gym.logger(
                "You are calling rend3er method without specifuing any render mode."
                "You can specify the render_mode at initialization"
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
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

        world_width = self.x_thereshold * 2
        scale = self.screen_width / world_width
        armwi_px = 20
        armlen_px = scale * self._length_arm

        if self.state is None:
            return None
        
        phi = self.state[0]
        cosphi = np.cos(phi)
        senphi = np.sin(phi)
        
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        center_x = self.screen_width / 2
        center_y = self.screen_height / 2 

        x1, y1 = 0, -armwi_px / 2          # Base inferior
        x2, y2 = 0, armwi_px / 2           # Base superior
        x3, y3 = armlen_px, armwi_px / 2   # Ponta superior
        x4, y4 = armlen_px, -armwi_px / 2  # Ponta inferior

        # 3. Aplicamos a Matriz de Rotação 2D
        p1_x = x1 * cosphi - y1 * senphi
        p1_y = x1 * senphi + y1 * cosphi

        p2_x = x2 * cosphi - y2 * senphi
        p2_y = x2 * senphi + y2 * cosphi

        p3_x = x3 * cosphi - y3 * senphi
        p3_y = x3 * senphi + y3 * cosphi

        p4_x = x4 * cosphi - y4 * senphi
        p4_y = x4 * senphi + y4 * cosphi

        c1 = (int(center_x + p1_x), int(center_y - p1_y))
        c2 = (int(center_x + p2_x), int(center_y - p2_y))
        c3 = (int(center_x + p3_x), int(center_y - p3_y))
        c4 = (int(center_x + p4_x), int(center_y - p4_y))
        
        arm_coords = [c1, c2, c3, c4]
        
        gfxdraw.aapolygon(self.surf, arm_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, arm_coords, (202, 152, 101))
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
    