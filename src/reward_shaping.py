"""
Reward shaping module for the Roberta robotic arm environment.
It defines the weighting constants and the core logic used to calculate 
the continuous and discrete rewards based on the agent's kinematics and actions.
"""

import numpy as np

# Reward and penalty weighting constants
PHI_WEIGHT = 5.0
SURVIVAL_REWARD = 1.0
PHI_DOT_WEIGHT = 1.0
ACTION_WEIGHT = 1.0
KILL_REWARD = -20000
PHI_WEIGHT_N = 5.0
LAMBDA = 0.001

def compute_reward(
    phi: float,
    phi_dot: float,
    setpoint: float,
    throttle: float,
    equilibrium: float,
    max_dangle: float,
    terminated: bool,
) -> float:
    """
    Calculates the shaped reward for the Roberta arm environment.

    Args:
        phi (float): The current angle or position of the arm.
        phi_dot (float): The current angular velocity of the arm.
        setpoint (float): The target angle/position the arm needs to reach.
        throttle (float): The current control action (motor usage) applied.
        equilibrium (float): The baseline throttle value required to maintain a steady state.
        max_dangle (float): The maximum allowed physical angle amplitude, used for normalization.
        terminated (bool): Flag indicating if a critical failure occurred (episode termination).

    Returns:
        float: The computed shaped reward for the current timestep.
    """

    # Extreme penalty for critical failure (e.g., arm angle exceeding allowed physical limits).
    if terminated:
        return float(KILL_REWARD)

    # Error calculation using L1 Norm (absolute value). 
    # This guarantees a constant correction gradient, forcing the agent to correct 
    # deviations even when it is millimetrically close to the target.
    # Metrics are normalized by dividing by the maximum amplitude limits (max_dangle).
    phi_dot_error = abs(phi_dot / (5 * max_dangle))
    phi_error = abs((phi - setpoint) / max_dangle)
    
    # Control effort penalty (motor usage). 
    # The reduced weight (multiplied by 0.1) ensures the agent focuses on positional 
    # accuracy (reaching the setpoint) rather than solely optimizing energy consumption.
    action_penalty = (ACTION_WEIGHT * 0.1) * abs(throttle - equilibrium)
    
    # Main composition: The agent earns a constant survival base reward and 
    # suffers deductions proportional to kinematic errors (angle and velocity) 
    # and the effort of the applied action. It also includes a Gaussian-like proximity bonus.
    reward = -(
        PHI_DOT_WEIGHT * phi_dot_error
        + PHI_WEIGHT * phi_error
        + action_penalty
    ) + (
        PHI_WEIGHT_N * np.exp(-phi_error**2 / LAMBDA)
        + SURVIVAL_REWARD
    )

    # Steady-state bonus (Target Zone):
    # Oversized reward condition triggered only when precision reaches less than 2% deviation, 
    # anchoring the arm to the exact setpoint location.
    if phi_error < 0.02: 
        reward += 5.0
    if phi_error < 0.005:
        reward += 5.0
        
    return float(reward)