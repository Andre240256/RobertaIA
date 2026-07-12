from stable3_aux_funcs import *
from stable_baselines3 import PPO

def create_ppo(env, lr, log_dir):
    r"""
    Initializes and configures a Proximal Policy Optimization (PPO) agent for the mechanical arm control task.
    
    The PPO algorithm optimizes the objective function:
    L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) ]
    
    This function sets up the neural architecture with shared or separate MLP layers for the 
    Actor (\pi) and Critic (V) networks.

    :param env: The environment to be trained on, compliant with gym.Env interface.
    :param lr: The initial learning rate for the policy/value networks.
    :param log_dir: String path to the directory where TensorBoard logs will be saved.
    :return: An initialized stable_baselines3.PPO model.
    """
    
    policy_keywards = dict(
        net_arch = dict(
            pi=[64],
            vf=[64]
        ),
        activation_fn=nn.ReLU
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_keywards,
        learning_rate=cossine_schedule(lr),
        device="cpu",
        gamma=0.99,
        batch_size=256,
        n_epochs=10,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=log_dir
    )

    return model