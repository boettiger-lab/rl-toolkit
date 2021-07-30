import argparse
import json
import os

import gym
import gym_conservation
import gym_fishing

# import **GYM OF INTEREST**
import optuna
import tensorflow as tf
from hyperparams_utils_sb2 import (
    sample_a2c_params,
    sample_acktr_params,
    sample_ppo2_params,
)
from simulate_vec_sb2 import simulate_mdp_vec
from stable_baselines import A2C, ACKTR, PPO2
from stable_baselines.common import make_vec_env

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# To avoid GPU memory hogging by TF
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-e", "--env", type=str, help="Environment name", default="conservation-v6"
)
parser.add_argument(
    "--study-name", type=str, help="Study name", default="trash"
)
parser.add_argument(
    "-a",
    "--algorithm",
    type=str,
    default="ppo2",
    help="Algorithm (a2c/acktr/ppo2)",
)
parser.add_argument(
    "--n-timesteps",
    type=int,
    default=int(2e6),
    help="Number of time steps for trainng",
)
parser.add_argument(
    "--env-kwargs",
    type=json.loads,
    help="Environment keyword arguments",
    default={
        "Tmax": 50,
    },
)
parser.add_argument(
    "--n-eval-episodes",
    type=int,
    help="Number of evaluation episodes",
    default=20,
)
parser.add_argument(
    "--n-trials", type=int, help="Number of trials", default=25
)
args = parser.parse_args()

algo_utils = {
    "a2c": (sample_a2c_params, A2C),
    "acktr": (sample_acktr_params, ACKTR),
    "ppo2": (sample_ppo2_params, PPO2),
}


def objective(trial):
    # Training the agent
    params, CustomLSTMPolicy = algo_utils[args.algorithm][0](trial)
    env = make_vec_env(
        lambda: gym.make(args.env, **args.env_kwargs),
        n_envs=params["n_envs"],
    )
    params.pop("n_envs")
    model = algo_utils[args.algorithm][1](
        CustomLSTMPolicy, env, verbose=0, **params
    )
    model.learn(total_timesteps=args.n_timesteps)
    # Evaluating the agent
    eval_env = gym.make(args.env)
    eval_df = simulate_mdp_vec(env, eval_env, model, args.n_eval_episodes)
    mean_rew = eval_df.groupby(["rep"]).sum().mean(axis=0)["reward"]

    return mean_rew


if __name__ == "__main__":
    # Creating an Optuna study that uses sqlite
    study_name = args.study_name  # Unique identifier of the study.
    storage_name = "sqlite:///tuning_studies/{}.db".format(study_name)
    # Sampling from hyperparameters using TPE over 50 trials
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        study_name=study_name,
        sampler=sampler,
        direction="maximize",
        storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)
    # Reporting best trial and making a quick plot to examine hyperparameters
    trial = study.best_trial
    print(f"Best hyperparams: {trial.params}")
    study.set_system_attr("algorithm", args.algorithm)
    study.set_system_attr("environment", args.env)
    study.set_system_attr("environment kwargs", args.env_kwargs)
    study.set_system_attr("num of timesteps", args.n_timesteps)
