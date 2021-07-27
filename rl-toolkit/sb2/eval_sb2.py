import gym
import gym_conservation
import gym_fishing
from gym_fishing.envs.shared_env import plot_mdp, simulate_mdp
from simulate_vec_sb2 import simulate_mdp_vec
from stable_baselines import A2C, ACKTR, PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import LstmPolicy

import argparse

# Argument parsing block; for help run `python tune_sb3.py -h`
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-name",
    type=str,
    help="Name of model to load",
    default="trash",
)
parser.add_argument(
    "-a",
    "--algorithm",
    type=str,
    default="ppo2",
    help="Algorithm (a2c/acktr/ppo2)",
)
parser.add_argument(
    "--n-envs",
    type=int,
    help="Number of environments to use in vec env",
    default=4,
)
parser.add_argument(
    "--image-name",
    type=str,
    help="Name of image to output",
    default="trash",
)
args = parser.parse_args()

algo_utils = {
    "a2c": (A2C,),
    "acktr": (ACKTR,),
    "ppo2": (PPO2,),
}

if __name__ == "__main__":
    env_kwargs = {
        "Tmax": 50,
    }
    model = algo_utils[args.algorithm.lower()][0].load(args.model_name)
    env = make_vec_env(
        args.environment, n_envs=args.n_envs, env_kwargs=env_kwargs
    )
    eval_env = gym.make(args.environment, **env_kwargs)
    plot_mdp(
        env,
        simulate_mdp_vec(env, eval_env, model, 10),
        output=f"{args.image_name}.png",
    )
