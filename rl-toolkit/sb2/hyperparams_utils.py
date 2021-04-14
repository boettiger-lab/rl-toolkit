from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import LstmPolicy
import gym
import gym_fishing


def sample_ppo2_params(trial):
    """
    Returns hyperparameter dicitonary to be passed to SB model
    """
    # Defining hyperparameters to sample over
    params = {
        'batch_size': trial.suggest_categorical('batch_size', [8, 64, 128, 516]),
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-5, 1),
        'n_steps': trial.suggest_categorical(
            'n_steps', [16, 32, 64, 128, 256, 512, 1024]),
        'gamma': trial.suggest_categorical('gamma', [0.9, 0.95, 0.99, 0.999, 1]),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-5, 1e1),
        'cliprange': trial.suggest_categorical('cliprange', [0.1, 0.2, 0.3, 0.4]),
        'noptepochs': trial.suggest_categorical('noptepochs', [1, 5, 10, 20, 30, 50]),
        'lam': trial.suggest_categorical('lambda', [0.8, 0.9, 0.95, 1.0]),
        'net_arch': trial.suggest_categorical('net_arch', ['small', 'med', 'large']),
        'n_lstm': trial.suggest_categorical('n_lstm', [1, 3, 25, 50, 100]),
        'n_envs': trial.suggest_categorical('n_envs', [4, 8, 16]),
    }
    # Following rl zoo
    if params['n_steps'] < params['batch_size']:
        nminibatches = 1
    else:
        nminibatches = 4
    # Mapping net_arch to actual network architectures for SB
    net_arch = {
        "small":  dict(pi=[64, 64], vf=[64, 64]),
        "med": dict(pi=[256, 256], vf=[256, 256]),
        "large": dict(pi=[400, 400], vf=[400, 400]),
    }[params['net_arch']]
    # Creating a custom LSTM policy using some of the hyperparameters selected above

    class CustomLSTMPolicy(LstmPolicy):
        def __init__(self, sess, ob_space, ac_space, n_env,
                     n_steps, n_batch, n_lstm=params['n_lstm'], reuse=False, **_kwargs):
            super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                             net_arch=[100, 'lstm', net_arch],
                             layer_norm=True, feature_extraction="mlp", **_kwargs)
    # Deleting keys that can't be used in SB models
    keys_to_delete = ['batch_size', 'n_lstm', 'net_arch']
    [params.pop(key) for key in keys_to_delete]
    # Adding keys that will be used in SB models
    params['nminibatches'] = nminibatches
    return params, CustomLSTMPolicy


def sample_a2c_params(trial):
    """
    Sampler for A2C hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    params = {
        'gamma': trial.suggest_categorical('gamma', [0.9, 0.95, 0.99, 0.999, 1]),
        'n_steps': trial.suggest_categorical(
            'n_steps', [16, 32, 64, 128, 256, 512, 1024, 2048]),
        'lr_schedule': trial.suggest_categorical('lr_schedule', ['linear', 'constant']),
        'learning_rate': trial.suggest_loguniform('lr', 1e-5, 1),
        'ent_coef': trial.suggest_uniform('ent_coef', 1e-3, 1e0),
        'vf_coef': trial.suggest_uniform('vf_coef', 0, 1),
        'n_lstm': trial.suggest_categorical('n_lstm', [1, 3, 25, 50, 100]),
        'net_arch': trial.suggest_categorical('net_arch', ['small', 'med', 'large']),
        'n_envs': trial.suggest_categorical('n_envs', [4, 8, 16]),
    }
    # Mapping net_arch to actual network architectures for SB
    net_arch = {
        "small":  dict(pi=[64, 64], vf=[64, 64]),
        "med": dict(pi=[256, 256], vf=[256, 256]),
        "large": dict(pi=[400, 400], vf=[400, 400]),
    }[params['net_arch']]
    # Creating a custom LSTM policy using some of the hyperparameters selected above

    class CustomLSTMPolicy(LstmPolicy):
        def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=params['n_lstm'], reuse=False, **_kwargs):
            super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                             net_arch=[100, 'lstm', net_arch],
                             layer_norm=True, feature_extraction="mlp", **_kwargs)
    # Deleting keys that can't be used in SB models
    keys_to_delete = ['n_lstm', 'net_arch']
    [params.pop(key) for key in keys_to_delete]

    return params, CustomLSTMPolicy


def sample_acktr_params(trial):
    """
    Sampler for ACKTR hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    params = {
        'gamma': trial.suggest_categorical('gamma', [0.9, 0.95, 0.99, 0.999, 1]),
        'n_steps': trial.suggest_categorical(
            'n_steps', [16, 32, 64, 128, 256, 512, 1024, 2048]),
        'lr_schedule': trial.suggest_categorical('lr_schedule', ['linear', 'constant']),
        'learning_rate': trial.suggest_loguniform('lr', 1e-5, 1),
        'ent_coef': trial.suggest_uniform('ent_coef', 1e-3, 1e0),
        'vf_coef': trial.suggest_uniform('vf_coef', 0, 1),
        'n_lstm': trial.suggest_categorical('n_lstm', [1, 3, 25, 50, 100]),
        'net_arch': trial.suggest_categorical('net_arch', ['small', 'med', 'large']),
        'n_envs': trial.suggest_categorical('n_envs', [4, 8, 16]),
    }
    # Mapping net_arch to actual network architectures for SB
    net_arch = {
        "small":  dict(pi=[64, 64], vf=[64, 64]),
        "med": dict(pi=[256, 256], vf=[256, 256]),
        "large": dict(pi=[400, 400], vf=[400, 400]),
    }[params['net_arch']]
    # Creating a custom LSTM policy using some of the hyperparameters selected above

    class CustomLSTMPolicy(LstmPolicy):
        def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=params['n_lstm'], reuse=False, **_kwargs):
            super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                             net_arch=[100, 'lstm', net_arch],
                             layer_norm=True, feature_extraction="mlp", **_kwargs)
    # Deleting keys that can't be used in SB models
    keys_to_delete = ['n_lstm', 'net_arch']
    [params.pop(key) for key in keys_to_delete]

    return params, CustomLSTMPolicy
