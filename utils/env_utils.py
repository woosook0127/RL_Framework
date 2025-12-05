"""Environment utilities for RL Framework"""
import gymnasium as gym
import numpy as np


def make_env_discrete(env_id, idx, capture_video, run_name, seed):
    """Create discrete action space environment"""
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env.reset(seed=seed + idx)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def make_env_continuous(env_id, idx, capture_video, run_name, gamma, seed):
    """Create continuous action space environment"""
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env.reset(seed=seed + idx)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        obs_space = env.observation_space
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), obs_space)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer with orthogonal weights"""
    import torch.nn
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

