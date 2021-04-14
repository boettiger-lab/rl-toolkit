import gym
import gym_fishing
from stable_baselines3 import SAC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def df_entry(df, env, rep, obs, action, reward, t):
    # Appending entry to the dataframe
    series = pd.Series([t, obs[0], action[0], reward, rep], index=df.columns)
    return df.append(series, ignore_index=True)


def simulate_mdp(env, model, n_eval_episodes):
    # A big issue with evaluating a vectorized environment is that SB automatically
    # resets an environment after a done flag.
    # To workaround this I have a single evaluation environment that I run
    # in parallel to the vectorized env.
    reps = int(n_eval_episodes)
    df = pd.DataFrame(columns=['time', 'state', 'action', 'reward', 'rep'])
    for rep in range(reps):
        # Creating the 2 environments
        obs = env.reset()
        # Initializing variables
        action = env.action_space.low
        reward = 0
        t = 0
        while True:
            df = df_entry(df, env, rep, obs, action, reward, t)
            t += 1
            # Using the vec env to do predictions
            action, state = model.predict(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break
        df = df_entry(df, env, rep, obs, action, reward, t)

    return df


def plot_mdp(df, output="results.png"):
    fig, axs = plt.subplots(3, 1)
    for i in np.unique(df.rep):
        results = df[df.rep == i]
        episode_reward = np.cumsum(results.reward)
        axs[0].plot(results.time, results.state, color="blue", alpha=0.3)
        axs[1].plot(results.time, results.action, color="blue", alpha=0.3)
        axs[2].plot(results.time, episode_reward, color="blue", alpha=0.3)

    axs[0].set_ylabel("state")
    axs[1].set_ylabel("action")
    axs[2].set_ylabel("reward")
    fig.tight_layout()
    plt.savefig(output)
    plt.close("all")
