from stable3_aux_funcs import  *
from stable_baselines3 import SAC


def create_sac(env, lr, log_dir):
    r"""
    Initializes and configures a Soft Actor-Critic (SAC) agent for the mechanical arm.
    
    SAC is an off-policy actor-critic algorithm that optimizes the maximum entropy objective:
    J(\theta) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} [r(s_t, a_t) + \alpha H(\pi(\cdot|s_t))]
    
    This implementation configures a symmetric Actor-Critic architecture and sets 
    the optimization parameters for stable learning in continuous action spaces.

    :param env: The environment to be trained on (e.g., RobertaEnv-v0).
    :param lr: The initial learning rate for the networks.
    :param log_dir: String path to the directory for TensorBoard logging.
    :return: An initialized stable_baselines3.SAC model.
    """
    
    policy_kwards =dict(
        net_arch = dict(
            pi=[64],
            qf=[64]
        ),
        activation_fn=nn.ReLU
    )
    

    model = SAC(
        "MlpPolicy",
        env=env,
        policy_kwargs=policy_kwards,
        device="cpu",
        gamma=0.99,
        ent_coef="auto",
        buffer_size=20000000,
        batch_size=256,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        learning_rate=linear_schedule(lr),
        verbose=1,
        tensorboard_log=log_dir
    )

    return model
