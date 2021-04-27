import gym
import gym_wildfire
from stable_baselines3.common.env_checker import check_env


def test_base_env():
    env = gym.make("wildfireCA-v0")
    check_env(env)
