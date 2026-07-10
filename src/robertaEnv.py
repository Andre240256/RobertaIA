import numpy as np
import math

import gymnasium as gym

from render import render
from reward_shaping import (
    ACTION_WEIGHT,
    KILL_REWARD,
    PHI_DOT_WEIGHT,
    PHI_WEIGHT,
    compute_reward,
)
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

        reward = compute_reward(
            phi=phi,
            phi_dot=phi_dot,
            setpoint=setpoint,
            throttle=throttle,
            equilibrium=self._equilibrium,
            max_dangle=self._max_dangle,
            terminated=terminated,
        )
        
        if self.render_mode == "human":
            self.render()

        self._steps += 1
        truncated = False
        if self._steps >= MAX_STEPS:
            truncated = True

        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

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
        render(self)
               
    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
    