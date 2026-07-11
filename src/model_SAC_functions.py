from stable3_aux_funcs import  *
from stable_baselines3 import SAC


def create_sac(env, lr, log_dir):

    
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
        buffer_size=10000000,
        batch_size=256,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        learning_rate=linear_schedule(lr),
        verbose=1,
        tensorboard_log=log_dir
    )

    return model
