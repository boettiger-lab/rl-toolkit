import optuna
from optuna.visualization import *
from stable_baselines3 import SAC, PPO, A2C, DDPG, TD3
from hyperparams_utils import *
import torch
from simulate_mdp import *
from simulate_vec import *
import gym
import gym_conservation
from stable_baselines3.common.env_util import make_vec_env
import argparse

# Argument parsing block; to get help on this from CL run `python tune_sb3.py -h`
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", type=str, help="Environment name", )
parser.add_argument("--study-name", type=str, help="Study name")
parser.add_argument("-a", type=str, default="ppo", help="Algorithm")
parser.add_argument("--n-timesteps", type=int, default=int(2e6),
                    help="Number of time steps for trainng")
parser.add_argument("--n-trials", type=int, default=int(25),
                    help="Number of tuning trials")
args = parser.parse_args()

# Handling the different parameter functions and models
algo_utils = {"ppo": (sample_ppo_params, PPO),
              "ddpg": (sample_ddpg_params, DDPG),
              "td3": (sample_td3_params, TD3),
              "sac": (sample_sac_params, SAC),
              "a2c": (sample_a2c_params, A2C), }


def objective(trial):
    # Getting the hyperparameters to test
    params,  policy_kwargs = algo_utils[args.a][0](trial)
    # Flag to keep track of whether using vectorized environment or not
    vec = False
    # Instatiating the environments
    try:
        env = make_vec_env(args.env, n_envs=params['n_envs'])
        params.pop('n_envs')
        vec = True
    except:
        env = gym.make(args.env)
    # Instatiating model and performing training
    model = algo_utils[args.a][1]("MlpPolicy", env, verbose=0,
                                  policy_kwargs=policy_kwargs, **params)
    model.learn(total_timesteps=int(args.n_timesteps))
    # Evaluating the agent and reporting the mean cumulative reward
    n_eval_episodes = 15
    if vec:
        eval_env = gym.make(args.env)
        eval_df = simulate_mdp_vec(env, eval_env, model, n_eval_episodes)
    else:
        eval_df = simulate_mdp(env, model, n_eval_episodes)
    mean_rew = eval_df.groupby(['rep']).sum().mean(axis=0)['reward']

    return mean_rew


if __name__ == "__main__":
    print("GPU Device: ", torch.cuda.current_device())
    # Creating an Optuna study that uses sqlite
    storage_name = f"sqlite:///studies/{args.study_name}.db"
    # Sampling from hyperparameters using TPE over 50 trials
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name=args.study_name, sampler=sampler,
                                direction='maximize', storage=storage_name,
                                load_if_exists=True)
    study.optimize(objective, n_trials=25)
    # Reporting best trial and making a quick plot to examine hyperparameters
    trial = study.best_trial
    fig = plot_contour(study)
    fig.write_image("trash.png")
    print(f"Best hyperparams: {trial.params}")
