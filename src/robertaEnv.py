import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


class RobertaEnv(gym.Env[np.ndarray, np.ndarray]):

    metadata = {
        "render_modes":["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: str | None = None):

        self._gravity = 9.8
        self._tau = 0.1
        self._throttle_converter = 100

        self._mass_arm = 1.0
        self._length_arm = 0.4
        self._inertia_momentum = (self._mass_arm * self._length_arm ** 2)/3

        self.kinematics_integrator = "euler"
        
        self._max_angle = 45
        self._min_angle = -52
        self._max_dangle = self._max_angle - self._min_angle

        self.x_thereshold = 2

        high = np.array(
            [
                self._max_angle,
                5 * self._max_dangle,
                self._max_angle,
                1,
            ],
            dtype=np.float32,
        )
        low = np.array(
            [
                self._min_angle,
                -5 * self._max_dangle,
                self._min_angle,
                0,
            ],
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(0, 1, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state: np.ndarray | None = None

        self.steps_beyond_terminated = None


    def step(self, action):
        assert self.action_space.contains(action), (
            f"{action!r} ({type(action)}) invalid"
        )
        assert self.state is not None, "Call reset before using step method."

        phi, phi_dot, setpoint, throttle = self.state

        motor_force = throttle * self._throttle_converter
        gravity_force = self._mass_arm * self._gravity

        cosphi = np.cos(phi)
        torque = (motor_force + 0.5 * gravity_force * cosphi) * self._length_arm
        phi_acc = torque / self._inertia_momentum

        if self.kinematics_integrator == "euler":
            phi = phi + self._tau * phi_dot
            phi_dot = phi_dot + self._tau * phi_acc
        else:
            phi_dot = phi_dot + self._tau * phi_acc
            phi = phi + self._tau * phi_dot
            
        self.state = np.array((phi, phi_dot, setpoint, throttle), dtype=np.float64)

        terminated = bool(
            phi < self._min_angle
            or phi > self._max_angle
        )

        phi_dot_error = (phi_dot/(5 * self._max_dangle))**2
        phi_error = (phi/self._max_dangle)**2
        PHI_DOT_WEIGHT = 0.5
        PHI_WEIGHT = 0.5

        if not terminated:
            reward = - (PHI_DOT_WEIGHT * phi_dot_error + PHI_WEIGHT * phi_error)
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
            reward = - (PHI_DOT_WEIGHT * phi_dot_error + PHI_WEIGHT * phi_error)
        else:
            if self.steps_beyond_terminated == 0:
                logger.warner(
                    "You are calling 'step()' even though this enviroment has alredy returned terminated = True"
                )
            self.steps_beyond_terminated += 1

            # calculare reward
            reward = 0.0
        
        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

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
        setpoint_inicial = self.np_random.uniform(
            low=self._min_angle, high=self._max_angle
        )
        throttle_inicial = self.np_random.uniform(
            low=0, high=1
        )

        self.state = np.array(
            [
                phi_inicial, phi_dot_inicial, setpoint_inicial, throttle_inicial
            ],
            dtype=np.float32
        )
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return self.state, {}
    
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
        armwi = 20
        armlen = scale * self._length_arm

        if self.state is None:
            return None
        
        phi = self.state[0]
        cosphi = np.cos(phi)
        senphi = np.sin(phi)
        
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        center_x = self.screen_width / 2
        center_y = self.screen_height / 2 

        p1_x = -senphi * (armwi / 2) * scale
        p1_y =  cosphi * (armwi / 2) * scale

        p2_x =  senphi * (armwi / 2) * scale
        p2_y =  cosphi * (armwi / 2) * scale  

        p3_x = (cosphi * armlen + senphi * armwi / 2) * scale
        p3_y = (senphi * armlen - cosphi * armwi / 2) * scale

        p4_x = (cosphi * armlen - senphi * armwi / 2) * scale
        p4_y = (senphi * armlen + cosphi * armwi / 2) * scale

        c1 = (int(float(center_x + p1_x)), int(float(center_y - p1_y)))
        c2 = (int(float(center_x + p2_x)), int(float(center_y - p2_y)))
        c3 = (int(float(center_x + p3_x)), int(float(center_y - p3_y)))
        c4 = (int(float(center_x + p4_x)), int(float(center_y - p4_y)))
        
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
    