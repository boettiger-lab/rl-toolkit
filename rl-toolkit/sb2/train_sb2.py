import optuna
from optuna.visualization import *

from simulate_vec_sb2 import simulate_mdp_vec
from train_utils import parse_hyperparams
from stable_baselines import A2C, ACKTR, PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import LstmPolicy

import json
import gym
import gym_conservation
import argparse
import tensorflow as tf

# Argument parsing block; to get help on this from CL run `python tune_sb3.py -h`
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", type=str, help="Environment name", default="conservation-v0")
parser.add_argument("--study-name", type=str, help="Study name", default="test")
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
args = parser.parse_args()

# Handling the different parameter functions and models
algo_utils = {
    "a2c": (A2C,),
    "acktr": (ACKTR,),
    "ppo2": (PPO2,),
}

def main():
  params, CustomLSTMPolicy, env = parse_hyperparams(args)
  # Instatiating model and performing training
  model = algo_utils[args.algorithm][0](CustomLSTMPolicy, env, verbose=2, **params)
  model.learn(total_timesteps=int(args.n_timesteps))
  # Evaluating the agent and reporting the mean cumulative reward
  eval_env = gym.make(args.env)
  eval_df = simulate_mdp_vec(env, eval_env, model, args.n_eval_episodes)

if __name__ =="__main__":
  main()
  
