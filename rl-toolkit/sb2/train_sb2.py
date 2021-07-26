import argparse
import json
import os

import gym
import gym_conservation
import tensorflow as tf
import yaml
from stable_baselines import A2C, ACKTR, PPO2
from train_utils import parse_hyperparams

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# To avoid GPU memory hogging by TF
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Argument parsing block; for help run `python tune_sb3.py -h`
parser = argparse.ArgumentParser()
parser.add_argument(
    "-e", "--env", type=str, help="Environment name", default="conservation-v0"
)
parser.add_argument(
    "--study-name", type=str, help="Study name", default="test"
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
    "--file",
    type=str,
    help="Name to save model as",
    default="trash",
)
args = parser.parse_args()

# Handling the different parameter functions and models
algo_utils = {
    "a2c": (A2C,),
    "acktr": (ACKTR,),
    "ppo2": (PPO2,),
}


def save_info(args, params):
    with open(f"models/{args.file}.yaml", "w") as file:
        params["environment"] = args.env
        params["algorithm"] = args.algorithm
        params["timesteps"] = args.n_timesteps
        params["environment_kwargs"] = args.env_kwargs
        documents = yaml.dump(params, file)


def main():
    params, CustomLSTMPolicy, env = parse_hyperparams(args)
    # Instatiating model and performing training
    model = algo_utils[args.algorithm][0](
        CustomLSTMPolicy, env, verbose=2, **params
    )
    model.learn(total_timesteps=int(args.n_timesteps))
    # Evaluating the agent and reporting the mean cumulative reward
    if not os.path.isdir("models/"):
        os.makedirs("models/")
    model.save(f"models/{args.file}")
    save_info(args, params)


if __name__ == "__main__":
    main()
