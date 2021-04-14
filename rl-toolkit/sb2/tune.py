import optuna
from optuna.visualization import *
from stable_baselines import PPO2, A2C, ACKTR
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import LstmPolicy
import gym
import gym_fishing
import gym_conservation
from simulate_vec import simulate_mdp_vec
from hyperparams_utils import sample_ppo2_params, sample_a2c_params, sample_acktr_params
import argparse
import tensorflow as tf

# To avoid GPU memory hogging by TF
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", type=str, help="Environment name")
parser.add_argument("--study-name", type=str, help="Study name")
parser.add_argument("-a", type=str, default="ppo2",
                    help="Algorithm (a2c/acktr/ppo2)")
parser.add_argument("--n-timesteps", type=int, default=int(2e6),
                    help="Number of time steps for trainng")
args = parser.parse_args()

algo_utils = {"a2c": (sample_a2c_params, A2C), "acktr": (
    sample_acktr_params, ACKTR), "ppo2": (sample_ppo2_params, PPO2)}[args.a]


def objective(trial):
    # Training the agent
    params,  CustomLSTMPolicy = algo_utils[0](trial)
    env_kwargs = {"Tmax": 50,
                  "alpha":  0.01,
                  }
    env = make_vec_env(
        args.env, n_envs=params['n_envs'], env_kwargs=env_kwargs)
    params.pop('n_envs')
    model = algo_utils[1](CustomLSTMPolicy, env, verbose=0, **params)
    model.learn(total_timesteps=args.n_timesteps)
    # Evaluating the agent and reporting the mean cumulative reward over 20 trials
    eval_env = gym.make(args.env)
    eval_df = simulate_mdp_vec(env, eval_env, model, 20)
    mean_rew = eval_df.groupby(['rep']).sum().mean(axis=0)['reward']

    return mean_rew


if __name__ == "__main__":
    # Creating an Optuna study that uses sqlite
    study_name = args.study_name  # Unique identifier of the study.
    storage_name = "sqlite:///tuning_studies/{}.db".format(study_name)
    # Sampling from hyperparameters using TPE over 50 trials
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name=study_name, sampler=sampler,
                                direction='maximize', storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=50)
    # Reporting best trial and making a quick plot to examine hyperparameters
    trial = study.best_trial
    print(f"Best hyperparams: {trial.params}")
