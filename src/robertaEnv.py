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
    """
    Implements a 1-Degree of Freedom (1-DOF) mechanical arm environment for Reinforcement Learning.
    The dynamics are governed by the equation of motion for a rigid body:
    I * \ddot{\phi} = \tau_{motor} - \tau_{gravity}
    
    Where:
    - \tau_{motor} = F_{motor} * L = (throttle * c_T) * L
    - \tau_{gravity} = F_{gravity} * (L/2) * \cos(\phi) = (m * g) * (L/2) * \cos(\phi)
    (Assuming the center of mass is at L/2).
    """

    metadata = {
        "render_modes":["human", "rgb_array"],
        "render_fps": 120,
    }

    def __init__(self, render_mode: str | None = None):
        """
        Initializes the RobertaEnv, defining the physical constants, constraints, and operational spaces.
        The state space is a continuous vector space R^3 representing [\phi, \dot{\phi}, \phi^*].
        The action space is a continuous vector space R^1 representing the normalized throttle.
        :param render_mode: string specifying the rendering method ("human" or "rgb_array").
        """
        self._gravity = 9.8
        self._tau = 0.01 
        self._throttle_converter = 12.25

        self._setpoint = None
        self._equilibrium = None
        self._mass_arm = None
        self._length_arm = None
        self._inertia_momentum = None

        self.kinematics_integrator = ""
        self._steps = 0
        
        self.x_thereshold = 2

        self._max_angle = 45.0 * math.pi / 180.0
        self._min_angle = -52.0 * math.pi / 180.0
        self._max_dangle = self._max_angle - self._min_angle

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
        """
        Executes one timestep of the environments dynamics using Euler or semi-implicit Euler integration.
        Calculates the physical interaction \ddot{\phi} = \Sigma \tau / I and updates the state.

        :param action: array of shape (1,) representing the continuous action chosen by the policy in [-1, 1].
        :return: tuple of (state, reward, terminated, truncated, info)
                 - state: np.ndarray of shape (3,) representing (\phi, \dot{\phi}, \phi^*).
                 - reward: float, scalar reward from the shaping function.
                 - terminated: bool, True if the arm exceeds angular limits.
                 - truncated: bool, True if MAX_STEPS is reached.
                 - info: dict, additional tracking metrics.
        """
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
        """
        Resets the environment to a random initial state and applies Domain Randomization
        to the digital twins physical parameters.

        :param seed: integer to seed the random number generator for reproducibility.
        :param options: dict with optional configurations.
        :return: tuple of (state, info)
                 - state: np.ndarray of shape (3,) representing (\phi_0, \dot{\phi}_0, \phi^*).
                 - info: dict with the sampled physical parameters and hyperparams.
        """
        super().reset(seed=seed)

        self._mass_arm, self._length_arm, self._inertia_momentum = self.sample_digitaltwin()

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
    
    def sample_digitaltwin(self) ->  tuple:
        """
        Samples the physical parameters for Domain Randomization.
        Models the arm as a uniform rod pivoting at one end, governed by I = (m * L^2) / 3.

        :return: tuple of floats (mass_arm, length_arm, inertia_momentum)
        """
        mass_arm = self.np_random.uniform(0.9, 1.1) #mass from 0.9 to 1.1
        length_arm = self.np_random.uniform(0.3, 0.5) #size is from 0.3m to 0.5m
        inertia_momentum = (mass_arm * length_arm ** 2)/3
        return mass_arm, length_arm, inertia_momentum
