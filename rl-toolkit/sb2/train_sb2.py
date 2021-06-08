import optuna
from optuna.visualization import *

from simulate_vec_sb2 import simulate_mdp_vec
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
parser.add_argument("-e", "--env", type=str, help="Environment name")
parser.add_argument("--study-name", type=str, help="Study name")
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
  storage_name = f"sqlite:///tuning_studies/{args.study_name}.db"
  study = optuna.load_study(study_name=args.study_name, storage=storage_name)
  trial = study.best_trial
  params = trial.params
  vec = False
  # Make the environment depending on number of environments in params
  try:
    env = make_vec_env(
        lambda: gym.make(args.env, **args.env_kwargs),
        n_envs=params["n_envs"],
    )
    params.pop('n_envs')
    vec = True
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
          **_kwargs
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
              **_kwargs
          )
          
  # Deleting keys that can't be used in SB models
  import pdb; pdb.set_trace()
  keys_to_delete = ["batch_size", "n_lstm", "net_arch", "joker"]
  if "lambda" in params:
    keys_to_delete.append("lambda")
    params['lam'] = params['lambda']
  [params.pop(key) for key in keys_to_delete if key in params]
  # Instatiating model and performing training
  model = algo_utils[args.algorithm][0](CustomLSTMPolicy, env, verbose=0, **params)
  model.learn(total_timesteps=int(args.n_timesteps))
  # Evaluating the agent and reporting the mean cumulative reward
  eval_env = gym.make(args.env)
  eval_df = simulate_mdp_vec(env, eval_env, model, args.n_eval_episodes)

if __name__ =="__main__":
  main()
  
