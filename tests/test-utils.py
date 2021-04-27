import os

import gym


def test_scripts():
    algorithms = ["a2c", "ddpg", "ppo", "sac", "td3"]
    envs = ["fishing-v1", "conservation-v2"]
    for env in envs:
        for algorithm in algorithms:
            os.system(
                f"python rl-toolkit/sb3/tune_sb3.py -a {algorithm} \
                        --study-name test_{algorithm} --n-trials 5 \
                        -e {env}"
            )
    os.system("rm -r studies/")
    os.system("rm render.csv")
