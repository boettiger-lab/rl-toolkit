import numpy as np
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

noise_dict = {
    "Normal": NormalActionNoise,
    "Ornstein": OrnsteinUhlenbeckActionNoise,
}


def sample_ppo_params(trial):
    """
    Returns hyperparameter dicitonary to be passed to SB model
    """
    # Defining hyperparameters to sample over
    params = {
        "batch_size": trial.suggest_categorical(
            "batch_size", [8, 64, 128, 516, 1024]
        ),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1),
        "n_steps": trial.suggest_categorical(
            "n_steps", [16, 32, 64, 128, 256, 512, 1024]
        ),
        "gamma": trial.suggest_categorical(
            "gamma", [0.9, 0.95, 0.99, 0.999, 1]
        ),
        "ent_coef": trial.suggest_loguniform("ent_coef", 1e-5, 1),
        "n_epochs": trial.suggest_categorical(
            "n_epochs", [1, 5, 10, 20, 30, 50]
        ),
        "net_arch": trial.suggest_categorical(
            "net_arch", ["small", "med", "large"]
        ),
        "n_envs": trial.suggest_categorical("n_envs", [4, 8, 16]),
        "log_std_init": trial.suggest_loguniform("log_std_init", 1e-3, 1e0),
    }
    # Following batch size handling on RL zoo
    if params["batch_size"] > params["n_steps"]:
        params["batch_size"] = params["n_steps"]
    # Mapping net_arch to actual network architectures for SB
    net_arch = {
        "small": dict(pi=[64, 64], qf=[64, 64]),
        "med": dict(pi=[256, 256], qf=[256, 256]),
        "large": dict(pi=[400, 400], qf=[400, 400]),
    }[params["net_arch"]]
    # Creating a custom LSTM policy
    policy_kwargs = dict(
        net_arch=[net_arch], log_std_init=params["log_std_init"]
    )
    # Deleting keys that can't be used in SB models
    keys_to_delete = ["net_arch", "log_std_init"]
    [params.pop(key) for key in keys_to_delete]

    return params, policy_kwargs


def sample_a2c_params(trial):
    """
    Returns hyperparameter dicitonary to be passed to SB model
    """
    # Defining hyperparameters to sample over
    params = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1),
        "n_steps": trial.suggest_categorical(
            "n_steps", [16, 32, 64, 128, 256, 512, 1024]
        ),
        "gamma": trial.suggest_categorical(
            "gamma", [0.9, 0.95, 0.99, 0.999, 1]
        ),
        "ent_coef": trial.suggest_loguniform("ent_coef", 1e-5, 1),
        "net_arch": trial.suggest_categorical(
            "net_arch", ["small", "med", "large"]
        ),
        "n_envs": trial.suggest_categorical("n_envs", [4, 8, 16]),
        "log_std_init": trial.suggest_loguniform("log_std_init", 1e-3, 1e0),
    }
    # Mapping net_arch to actual network architectures for SB
    net_arch = {
        "small": dict(pi=[64, 64], qf=[64, 64]),
        "med": dict(pi=[256, 256], qf=[256, 256]),
        "large": dict(pi=[400, 400], qf=[400, 400]),
    }[params["net_arch"]]
    # Creating a custom LSTM policy
    policy_kwargs = dict(
        net_arch=[net_arch], log_std_init=params["log_std_init"]
    )
    # Deleting keys that can't be used in SB models
    keys_to_delete = ["net_arch", "log_std_init"]
    [params.pop(key) for key in keys_to_delete]

    return params, policy_kwargs


def sample_sac_params(trial):
    """
    Returns hyperparameter dicitonary to be passed to SB model
    """
    # Defining hyperparameters to sample over
    params = {
        "batch_size": trial.suggest_categorical(
            "batch_size", [8, 64, 128, 516, 1024]
        ),
        "buffer_size": trial.suggest_categorical(
            "buffer_size", [int(1e4), int(1e5), int(1e6)]
        ),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1),
        "gamma": trial.suggest_categorical(
            "gamma", [0.9, 0.95, 0.99, 0.999, 1]
        ),
        "ent_coef": trial.suggest_loguniform("ent_coef", 1e-5, 1),
        "net_arch": trial.suggest_categorical(
            "net_arch", ["small", "med", "large"]
        ),
        "log_std_init": trial.suggest_loguniform("log_std_init", 1e-3, 1e0),
        "n_envs": 1,
        "seed": trial.suggest_categorical("seed", [2, 8, 16, 24, 32]),
    }
    # Mapping `net_arch` to actual network architectures
    net_arch = {
        "small": dict(pi=[64, 64], qf=[64, 64]),
        "med": dict(pi=[256, 256], qf=[256, 256]),
        "large": dict(pi=[400, 400], qf=[400, 400]),
    }[params["net_arch"]]
    # Creating a custom LSTM policy
    policy_kwargs = dict(
        net_arch=net_arch, log_std_init=params["log_std_init"]
    )
    # Deleting keys that can't be used in SB models
    keys_to_delete = ["net_arch", "log_std_init"]
    [params.pop(key) for key in keys_to_delete]

    return params, policy_kwargs


def sample_td3_params(trial):
    """
    Returns hyperparameter dicitonary to be passed to SB model
    """
    # Defining hyperparameters to sample over
    params = {
        "batch_size": trial.suggest_categorical(
            "batch_size", [8, 64, 128, 516, 1024]
        ),
        "buffer_size": trial.suggest_categorical(
            "buffer_size", [int(1e4), int(1e5), int(1e6)]
        ),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1),
        "gamma": trial.suggest_categorical(
            "gamma", [0.9, 0.95, 0.99, 0.999, 1]
        ),
        "net_arch": trial.suggest_categorical(
            "net_arch", ["small", "med", "large"]
        ),
        "noise_std": trial.suggest_uniform("noise_std", 0, 0.2),
        "noise_type": trial.suggest_categorical(
            "noise_type", ["Normal", "Ornstein"]
        ),
        "n_envs": 1,
    }
    # Mapping net_arch to actual network architectures for SB
    net_arch = {
        "small": dict(pi=[64, 64], qf=[64, 64]),
        "med": dict(pi=[256, 256], qf=[256, 256]),
        "large": dict(pi=[400, 400], qf=[400, 400]),
    }[params["net_arch"]]
    policy_kwargs = dict(net_arch=net_arch)
    # Handling action noise -- **
    action_noise = noise_dict[params["noise_type"]](
        mean=np.zeros(1), sigma=params["noise_std"] * np.ones(1)
    )
    params["action_noise"] = action_noise
    # Deleting keys that can't be used in SB models
    keys_to_delete = ["net_arch", "noise_std", "noise_type"]
    [params.pop(key) for key in keys_to_delete]

    return params, policy_kwargs


def sample_ddpg_params(trial):
    """
    Returns hyperparameter dicitonary to be passed to SB model
    """
    # Defining hyperparameters to sample over
    params = {
        "batch_size": trial.suggest_categorical(
            "batch_size", [8, 64, 128, 516, 1024]
        ),
        "buffer_size": trial.suggest_categorical(
            "buffer_size", [int(1e4), int(1e5), int(1e6)]
        ),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1),
        "gamma": trial.suggest_categorical(
            "gamma", [0.9, 0.95, 0.99, 0.999, 1]
        ),
        "net_arch": trial.suggest_categorical(
            "net_arch", ["small", "med", "large"]
        ),
        "noise_std": trial.suggest_uniform("noise_std", 0, 0.2),
        "noise_type": trial.suggest_categorical(
            "noise_type", ["Normal", "Ornstein"]
        ),
        "n_envs": 1,
    }
    # Mapping net_arch to actual network architectures for SB
    net_arch = {
        "small": dict(pi=[64, 64], qf=[64, 64]),
        "med": dict(pi=[256, 256], qf=[256, 256]),
        "large": dict(pi=[400, 400], qf=[400, 400]),
    }[params["net_arch"]]
    policy_kwargs = dict(net_arch=net_arch)
    # Handling action noise -- **
    action_noise = noise_dict[params["noise_type"]](
        mean=np.zeros(1), sigma=params["noise_std"] * np.ones(1)
    )
    params["action_noise"] = action_noise
    # Deleting keys that can't be used in SB models
    keys_to_delete = ["net_arch", "noise_std", "noise_type"]
    [params.pop(key) for key in keys_to_delete]

    return params, policy_kwargs


def sample_dqn_params(trial):
    """
    Returns hyperparameter dicitonary to be passed to SB model
    """
    # Defining hyperparameters to sample over
    params = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1),
        "gamma": trial.suggest_categorical(
            "gamma", [0.9, 0.95, 0.99, 0.999, 1]
        ),
        "exploration_fraction": trial.suggest_uniform(
            "exploration_fraction", 0, 0.5
        ),
        "exploration_final_eps": trial.suggest_uniform(
            "exploration_final_eps", 0, 0.2
        ),
        "net_arch": trial.suggest_categorical(
            "net_arch", ["small", "med", "large"]
        ),
        "n_envs": 1,
        "target_update_interval": trial.suggest_categorical(
            "target_update_interval", [1, 1000, 5000, 10000, 15000, 20000]
        ),
        "train_freq": trial.suggest_categorical(
            "train_freq", [1, 4, 8, 16, 128, 256, 1000]
        ),
        "subsample_steps": trial.suggest_categorical(
            "subsample_steps", [1, 2, 4, 8]
        ),
    }
    params["gradient_steps"] = max(
        params["train_freq"] // params["subsample_steps"], 1
    )
    # Mapping net_arch to actual network architectures for SB
    net_arch = {
        "small": [64, 64],
        "med": [256, 256],
        "large": [400, 400],
    }[params["net_arch"]]
    policy_kwargs = dict(net_arch=net_arch)
    # Deleting keys that can't be used in SB models
    keys_to_delete = ["net_arch", "train_freq", "subsample_steps"]
    [params.pop(key) for key in keys_to_delete]

    return params, policy_kwargs
