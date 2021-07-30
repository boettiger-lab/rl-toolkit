import pandas as pd


def df_entry_vec(df, env, rep, obs, action, reward, t):
    # Appending entry to the dataframe
    series = pd.Series(
        [
            t,
            env.get_unscaled_state(obs[0]),
            env.get_unscaled_action(action[0]),
            reward[0],
            rep,
        ],
        index=df.columns,
    )
    return df.append(series, ignore_index=True)


def simulate_mdp_vec(env, eval_env, model, n_eval_episodes):
    # A big issue with evaluating a vectorized environment is that
    # SB automatically resets an environment after a done flag.
    # To workaround this I have a single evaluation environment that
    # I run in parallel to the vectorized env.
    reps = int(n_eval_episodes)
    df = pd.DataFrame(columns=["time", "state", "action", "reward", "rep"])
    for rep in range(reps):
        # Creating the 2 environments
        e_obs = eval_env.reset()
        obs = env.reset()
        obs[0] = e_obs  # Passing first obs from eval env into first index
        # Initializing variables
        state = None
        done = [False for _ in range(env.num_envs)]
        action = [[env.action_space.low[0]] for _ in range(env.num_envs)]
        reward = [0 for _ in range(env.num_envs)]
        t = 0
        while True:
            df = df_entry_vec(df, eval_env, rep, obs, action, reward, t)
            t += 1
            # Using the vec env to do predictions
            action, state = model.predict(
                obs, state=state, mask=done, deterministic=True
            )
            obs, reward, done, info = env.step(action)
            # Stepping the eval env along with the vec env
            e_obs, e_reward, e_done, e_info = eval_env.step(action[0])
            # Passing the evaluation env in for the first vec env's
            # observations. This is to avoid automatic resetting when
            # `done=True` which is a constraint of vectorized environments.
            # Unfortunately, a recurrent trained agent must be evaluated on
            # the number of vectorized envs it was trained on.
            obs[0] = e_obs
            if e_done:
                break
        df = df_entry_vec(df, eval_env, rep, obs, action, reward, t)

    return df
