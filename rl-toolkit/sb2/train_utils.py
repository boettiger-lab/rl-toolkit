import gym
import optuna
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import LstmPolicy


def parse_hyperparams(args):
    storage_name = f"sqlite:///tuning_studies/{args.study_name}.db"
    study = optuna.load_study(study_name=args.study_name, storage=storage_name)
    trial = study.best_trial
    params = trial.params
    # Make the environment depending on number of environments in params
    try:
        env = make_vec_env(
            lambda: gym.make(args.env, **args.env_kwargs),
            n_envs=params["n_envs"],
        )
        params.pop("n_envs")
    except:
        env = gym.make(args.env)
    # Constructing the network architecture
    # Mapping net_arch to actual network architectures for SB
    net_arch = {
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "med": dict(pi=[256, 256], vf=[256, 256]),
        "large": dict(pi=[400, 400], vf=[400, 400]),
    }[params["net_arch"]]
    # Creating a custom LSTM policy

    class CustomLSTMPolicy(LstmPolicy):
        def __init__(
            self,
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            n_lstm=params["n_lstm"],
            reuse=False,
            **_kwargs,
        ):
            super().__init__(
                sess,
                ob_space,
                ac_space,
                n_env,
                n_steps,
                n_batch,
                n_lstm,
                reuse,
                net_arch=[100, "lstm", net_arch],
                layer_norm=True,
                feature_extraction="mlp",
                **_kwargs,
            )

    # Deleting keys that can't be used in SB models
    keys_to_delete = ["batch_size", "n_lstm", "net_arch", "joker"]
    if "lambda" in params:
        keys_to_delete.append("lambda")
        params["lam"] = params["lambda"]
    [params.pop(key) for key in keys_to_delete if key in params]
    return params, CustomLSTMPolicy, env
