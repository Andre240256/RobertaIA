import numpy as np
import math

import gymnasium as gym

from render import render
from edo_solver import edo_solver
from reward_shaping import (
    ACTION_WEIGHT,
    KILL_REWARD,
    PHI_DOT_WEIGHT,
    PHI_WEIGHT,
    compute_reward,
)
MAX_STEPS = 2000


class RobertaEnv(gym.Env[np.ndarray, np.ndarray]):
    r"""
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
        "render_fps": 100,
    }

    def __init__(self, render_mode: str | None = None):
        r"""
        Initializes the RobertaEnv, defining the physical constants, constraints, and operational spaces.
        The state space is a continuous vector space R^3 representing [\phi, \dot{\phi}, \phi^*].
        The action space is a continuous vector space R^1 representing the normalized throttle.
        :param render_mode: string specifying the rendering method ("human" or "rgb_array").
        """
        self._gravity = 9.81
        self._tau = 0.01 

        self._throttle_converter = None
        self._setpoint = None
        self._equilibrium = None
        self._mass_arm = None
        self._length_arm = None
        self._inertia_momentum = None
        self._friction_constant = None
        self._edo_solver = None

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
                1.0,
            ],
            dtype=np.float32,
        )
        low = np.array(
            [
                self._min_angle,
                -5 * self._max_dangle,
                self._min_angle,
                -1.0,
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
        r"""
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

        phi, phi_dot, setpoint, _ = self.state

        throttle = 0.5 * float(action[0]) + 0.5
        if throttle < 0.05:
            throttle = 0.0 #Ponto de zona morta

        # store last throttle for rendering
        self._last_throttle = float(throttle)

        motor_force = throttle * self._throttle_converter
        torque_motor = motor_force * self._length_arm 

        Yi = np.array([phi, phi_dot])
        Yprox = self._edo_solver.passoRK4(0, Yi, torque_motor)
        phi, phi_dot = Yprox[0], Yprox[1]
        
            
        terminated = bool(
            phi < self._min_angle
            or phi > self._max_angle
        )

        # Keep observations within bounds even on termination.
        self._prev_action = float(action[0])
        raw_state = np.array((phi, phi_dot, setpoint, self._prev_action), dtype=np.float64)
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
        r"""
        Resets the environment to a random initial state and applies Domain Randomization
        to the digital twins physical parameters.

        :param seed: integer to seed the random number generator for reproducibility.
        :param options: dict with optional configurations.
        :return: tuple of (state, info)
                 - state: np.ndarray of shape (3,) representing (\phi_0, \dot{\phi}_0, \phi^*).
                 - info: dict with the sampled physical parameters and hyperparams.
        """
        super().reset(seed=seed)

        self._mass_arm = None
        self._length_arm = None
        self._inertia_momentum = None
        self._friction_constant = None
        self._throttle_converter = None

        if options is not None:
            self._mass_arm = options["mass_arm"]
            self._length_arm = options["length_arm"]
            self._inertia_momentum = (options["inertia_momentum"] if 'inertia_momentum' in options
                                      else (self._mass_arm * self._length_arm**2)/3) 
            self._friction_constant = options["friction_constant"]
            self._throttle_converter = options['throttle_converter']
        
        self._sample_digitaltwin()
            
        self._edo_solver = edo_solver(self._edo_func(), self._tau)

        phi_inicial = self._last_phi_obs = self.np_random.uniform(
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

        self._prev_action = 0.0
        self.state = np.array(
            [
                phi_inicial, phi_dot_inicial, self._setpoint, self._prev_action
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
    
    def _sample_digitaltwin(self):
        r"""
        Samples the physical parameters for Domain Randomization.
        Models the arm as a uniform rod pivoting at one end, governed by I = (m * L^2) / 3.

        :return: tuple of floats (mass_arm, length_arm, inertia_momentum)
        """
        #mass from 0.9 to 1.1
        self._mass_arm = self.np_random.uniform(0.9, 1.1) if self._mass_arm is None else self._mass_arm 
        #size is from 0.3m to 0.5m
        self._length_arm = self.np_random.uniform(0.3, 0.5) if self._length_arm is None else self._length_arm
        self._inertia_momentum = (self._mass_arm * self._length_arm ** 2)/3 if self._inertia_momentum is None else self._inertia_momentum
        #friction_constant from 0 to 1.0
        self._friction_constant = self.np_random.uniform(0.01, 0.25) if self._friction_constant is None else self._friction_constant
        #throttle converter from 7.5 to 20
        self._throttle_converter = self.np_random.uniform(12.0, 12.5) if self._throttle_converter is None else self._throttle_converter

    def _edo_func(self):
        def F(t, Y, torque_motor):
            phi = Y[0]
            phi_dot = Y[1]

            Yprox = np.zeros(2)
            Yprox[0] = phi_dot
            
            torque_gravidade = (self._mass_arm * self._gravity * 
                                np.cos(phi) * self._length_arm)
            torque_atrito = self._friction_constant * phi_dot

            Yprox[1] = (torque_motor - torque_gravidade - torque_atrito)/self._inertia_momentum

            return Yprox
        return F 
