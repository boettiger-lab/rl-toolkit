import argparse
import os

import gym
# import **PUT GYM OF INTEREST HERE**
import numpy as np
import optuna
import yaml
from hyperparams_utils import noise_dict
from simulate_vec import plot_mdp, simulate_mdp_vec
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env

# Argument parsing block; to get help here run `python tune_sb3.py -h`
parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--env",
    type=str,
    default="conservation-v2",
    help="Environment name",
)
parser.add_argument(
    "--study-name", type=str, default="a2c_04_09_21", help="Study name"
)
parser.add_argument(
    "-a",
    "--algorithm",
    type=str,
    default="ppo",
    help="Algorithm (e.g. 'sac', 'ppo')",
)
parser.add_argument(
    "--n-timesteps",
    type=int,
    default=int(1e4),
    help="Number of time steps for trainng",
)
parser.add_argument("--selected", action="store_true")
parser.add_argument(
    "--n-eval-episodes",
    default=10,
    type=int,
    help="Number of evaluation episodes to test agent",
)
parser.add_argument(
    "--save", action="store_true", help="Whether to save model"
)
parser.add_argument(
    "--output-name",
    type=str,
    default="trash",
    help="What file name to use for all outputs",
)
parser.add_argument(
    "--plot", action="store_true", help="Choose to ouput a png plot of SAR"
)
args = parser.parse_args()
args.algorithm = args.algorithm.lower()

# Handling the different parameter functions and models
algo_utils = {
    "ppo": PPO,
    "ddpg": DDPG,
    "td3": TD3,
    "sac": SAC,
    "a2c": A2C,
}


def main():
    # In the first if then block, fetching hyperparameters either
    # from a yaml in the hyperparams directory or from the best trial
    # in the specified study.
    if args.selected:
        assert os.path.exists("hyperparams"), "No hyperparams directory found."
        assert os.path.exists(
            f"hyperparams/{args.env}.yaml"
        ), "No yaml file found for this environment."
        with open(f"hyperparams/{args.env}.yaml") as file:
            params = yaml.load(file, Loader=yaml.FullLoader)[args.algorithm]
    else:
        storage_name = f"sqlite:///studies/{args.study_name}.db"
        study = optuna.load_study(
            study_name=args.study_name, storage=storage_name
        )
        if "algorithm" in study.system_attrs:
            assert (
                args.algorithm == study.system_attrs["algorithm"]
            ), "Algorithm used in study and for training are not the same"
        trial = study.best_trial
        params = trial.params

    # Make the environment depending on number of environments in params.
    # Default is to use one environment
    if "n_envs" in params:
        env = make_vec_env(args.env, n_envs=params["n_envs"])
        params.pop("n_envs")
    else:
        env = make_vec_env(args.env, n_envs=1)

    keys_to_delete = []
    policy_kwargs = None
    # Handling the network architecture which is passed to the model as
    # policy_kwargs
    if "net_arch" in params:
        net_arch = {
            "small": dict(pi=[64, 64], qf=[64, 64]),
            "med": dict(pi=[256, 256], qf=[256, 256]),
            "large": dict(pi=[400, 400], qf=[400, 400]),
        }[params["net_arch"]]
        if args.algorithm in ["ppo", "a2c"]:
            policy_kwargs = dict(
                net_arch=[net_arch], log_std_init=params["log_std_init"]
            )
            keys_to_delete = ["net_arch", "log_std_init"]
        elif args.algorithm == "sac":
            policy_kwargs = dict(
                net_arch=net_arch, log_std_init=params["log_std_init"]
            )
            keys_to_delete = ["net_arch", "log_std_init"]
        else:
            policy_kwargs = dict(net_arch=net_arch)
            keys_to_delete = ["net_arch"]
    # Handling action noise if it is given in the params dict
    if "noise_type" in params:
        action_noise = noise_dict[params["noise_type"]](
            mean=np.zeros(1), sigma=params["noise_std"] * np.ones(1)
        )
        params["action_noise"] = action_noise
        keys_to_delete.extend(("noise_std", "noise_type"))
    # Handling noptepochs which I erroneously labeled in an early study.
    if "noptepochs" in params:
        params["n_epochs"] = params["noptepochs"]
        keys_to_delete.append("noptepochs")
    # Deleting parameters that will throw errors when unpacked
    [params.pop(key) for key in keys_to_delete]
    # Instatiating model and performing training
    model = algo_utils[args.algorithm](
        "MlpPolicy", env, verbose=2, policy_kwargs=policy_kwargs, **params
    )
    model.learn(total_timesteps=int(args.n_timesteps))
    # Saving model if flagged
    if args.save:
        model.save(f"{args.ouput_name}")
    # Evaluating the agent and reporting the mean cumulative reward
    eval_env = gym.make(args.env)
    eval_df = simulate_mdp_vec(env, eval_env, model, args.n_eval_episodes)
    # Plotting if flagged
    if args.plot:
        plot_mdp(eval_df, output=f"{args.output_name}.png")


if __name__ == "__main__":
    main()
