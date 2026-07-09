PHI_WEIGHT = 5.0
SURVIVAL_REWARD = 1.0
PHI_DOT_WEIGHT = 1.0
ACTION_WEIGHT = 1.0
KILL_REWARD = -20000


def compute_reward(
    phi: float,
    phi_dot: float,
    setpoint: float,
    throttle: float,
    equilibrium: float,
    max_dangle: float,
    terminated: bool,
) -> float:
    """Compute the shaping reward for the Roberta arm environment."""

    if terminated:
        return float(KILL_REWARD)

    phi_dot_error = (phi_dot / (5 * max_dangle)) ** 2
    phi_error = ((phi - setpoint) / max_dangle) ** 2
    reward = -(
        PHI_DOT_WEIGHT * phi_dot_error
        + PHI_WEIGHT * phi_error
        + ACTION_WEIGHT * (throttle - equilibrium) ** 2
    ) + SURVIVAL_REWARD
    return float(reward)