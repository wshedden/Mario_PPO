import numpy as np
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import gym_super_mario_bros
from utils import SaveOnBestTrainingRewardCallback


def load_model_for_training(model_name, env_id, skip, num_cpu, device, learning_rate):
    # this function loads the model from the file or creates a new one
    # check if the model name is a valid file name
    if os.path.isfile(model_name):
        # load the model from the file
        env = VecMonitor(SubprocVecEnv([make_env(env_id, i, skip=skip) for i in range(num_cpu)]), 'tmp/monitor')
        model = PPO.load(model_name, env=env, device=device)
        print("Model loaded from file.")
    elif model_name == "new":
        # create a default model with some parameters
        env = VecMonitor(SubprocVecEnv([make_env(env_id, i, skip=skip) for i in range(num_cpu)]), 'tmp/monitor')
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_super_mario_bros_tensorboard/", learning_rate=learning_rate, device=device)
        print("New model created.")
    else:
        # raise an error if the model name is invalid
        raise ValueError(f"Invalid model name: {model_name}")
    return model


def make_env(env_id, rank, seed=0, skip=1):
    def _init():
        env = gym_super_mario_bros.make(env_id)
        env = MaxAndSkipEnv(env, skip=skip)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def train_model(model, log_dir, learning_rate):
    # this function trains the model using the given environment and log directory
    print("Training...")
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    log_name = "ppo_smb_lr_" + str(learning_rate)
    model.learn(total_timesteps=10000000, callback=callback, tb_log_name=log_name)
    model.save("ppo_super_mario_bros")
    print("Done!")






