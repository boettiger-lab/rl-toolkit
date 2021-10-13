import argparse
import os

import gym
import gym_conservation
import gym_fishing
import gym_climate
import optuna
import torch
from hyperparams_utils import (
    sample_a2c_params,
    sample_ddpg_params,
    sample_ppo_params,
    sample_sac_params,
    sample_td3_params,
)
from simulate_vec import simulate_mdp_vec
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env

# Argument parsing block; to get help here run `python tune_sb3.py -h`
parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--env",
    default="conservation-v2",
    type=str,
    help="Environment name",
)
parser.add_argument(
    "--study-name", default="study", type=str, help="Study name"
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
    default=int(1e1),
    help="Number of time steps for trainng",
)
parser.add_argument(
    "--n-trials", type=int, default=int(25), help="Number of tuning trials"
)
parser.add_argument(
    "--n-eval-episodes",
    type=int,
    default=int(10),
    help="Number of evaluation episodes",
)
args = parser.parse_args()
args.algorithm = args.algorithm.lower()

# Handling the different parameter functions and models
algo_utils = {
    "ppo": (sample_ppo_params, PPO),
    "ddpg": (sample_ddpg_params, DDPG),
    "td3": (sample_td3_params, TD3),
    "sac": (sample_sac_params, SAC),
    "a2c": (sample_a2c_params, A2C),
}


def objective(trial):
    # Getting the hyperparameters to test
    params, policy_kwargs = algo_utils[args.algorithm][0](trial)
    # Flag to keep track of whether using vectorized environment or not
    # Instatiating the environments
    env = make_vec_env(args.env, n_envs=params["n_envs"])
    params.pop("n_envs")
    # Instatiating model and performing training
    model = algo_utils[args.algorithm][1](
        "MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs, **params
    )
    model.learn(total_timesteps=int(args.n_timesteps))
    # Evaluating the agent and reporting the mean cumulative reward
    eval_env = gym.make(args.env)
    eval_df = simulate_mdp_vec(env, eval_env, model, args.n_eval_episodes)
    mean_rew = eval_df.groupby(["rep"]).sum().mean(axis=0)["reward"]
    del model

    return mean_rew


if __name__ == "__main__":
    if not os.path.exists("studies"):
        os.makedirs("studies")
    # Fix this so it works in general case
    if torch.cuda.is_available():
        print("Active GPU Device: ", torch.cuda.current_device())
    # Creating an Optuna study that uses sqlite
    storage_name = f"sqlite:///studies/{args.study_name}.db"
    # Sampling from hyperparameters using TPE over 50 trials
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        study_name=args.study_name,
        sampler=sampler,
        direction="maximize",
        storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)
    # Saving attributes so we can retrace what is done for a .db
    study.set_system_attr("algorithm", args.algorithm)
    study.set_system_attr("environment", args.env)
    # Reporting best trial and making a quick plot to examine hyperparameters
    trial = study.best_trial
    print(f"Best hyperparams: {trial.params}")
