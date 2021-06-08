import optuna
from optuna.visualization import *
from hyperparams_utils import *

from simulate_vec_sb2 import simulate_mdp_vec
from stable_baselines import A2C, ACKTR, PPO2
from stable_baselines.common import make_vec_env

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
                             help="Number of time steps for trainng")
args = parser.parse_args()

# Handling the different parameter functions and models
algo_utils = {
    "a2c": (A2C,),
    "acktr": (ACKTR,),
    "ppo2": (PPO2,),
}

def main():
  storage_name = f"sqlite:///studies/{args.study_name}.db"
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
  net_arch = {
        "small":  dict(pi=[64, 64], qf=[64, 64]),
        "med": dict(pi=[256, 256], qf=[256, 256]),
        "large": dict(pi=[400, 400], qf=[400, 400]),
    }[params['net_arch']]
  if args.a in ["ppo", "sac", "a2c"]:
    if args.a in ['ppo', 'a2c']:
      policy_kwargs = dict(net_arch=[net_arch], log_std_init=params['log_std_init'])
    else:
      policy_kwargs = dict(net_arch=net_arch, log_std_init=params['log_std_init'])
    keys_to_delete = ['net_arch', 'log_std_init']
  else:
    policy_kwargs = dict(net_arch=net_arch)
    keys_to_delete = ['net_arch', 'noise_type', 'noise_std']
    action_noise = noise_dict[params['noise_type']](mean=np.zeros(1), 
                                         sigma=params['noise_std']*np.ones(1))
  if args.a == "ppo":
    params['n_epochs'] = params['noptepochs']
    keys_to_delete.append('noptepochs')
  [params.pop(key) for key in keys_to_delete]
  # Instatiating model and performing training
  model = algo_utils[args.a]("MlpPolicy", env, verbose=0, 
                                 policy_kwargs=policy_kwargs, **params)
  model.learn(total_timesteps=int(args.n_timesteps))
  # Evaluating the agent and reporting the mean cumulative reward
  n_eval_episodes = 15
  if vec:
      eval_env = gym.make(args.env)
      eval_df = simulate_mdp_vec(env, eval_env, model, n_eval_episodes)
  else:
      eval_df = simulate_mdp(env, model, n_eval_episodes)
  plot_mdp(eval_df, output=f"trash_{args.study_name}.png")

if __name__ =="__main__":
  main()
  
