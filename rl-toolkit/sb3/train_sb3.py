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
parser.add_argument("-e", "--env", type=str,
                    default='conservation-v2', help="Environment name", )
parser.add_argument("--study-name", type=str,
                    default="a2c_04_09_21", help="Study name")
parser.add_argument("-a", type=str, default="ppo", help="Algorithm")
parser.add_argument("--n-timesteps", type=int, default=int(5e5),
                    help="Number of time steps for trainng")
args = parser.parse_args()

# Handling the different parameter functions and models
algo_utils = {"ppo": PPO,
              "ddpg": DDPG,
              "td3": TD3,
              "sac": SAC,
              "a2c": A2C, }


def main():
    storage_name = f"sqlite:///studies/{args.study_name}.db"
    study = optuna.load_study(study_name=args.study_name, storage=storage_name)
    trial = study.best_trial
    params = trial.params
    vec = False
    # Make the environment depending on number of environments in params
    try:
        env = make_vec_env(args.env, n_envs=params['n_envs'])
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
            policy_kwargs = dict(
                net_arch=[net_arch], log_std_init=params['log_std_init'])
        else:
            policy_kwargs = dict(
                net_arch=net_arch, log_std_init=params['log_std_init'])
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


if __name__ == "__main__":
    main()
