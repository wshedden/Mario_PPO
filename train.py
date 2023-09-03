import numpy as np
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import gym_super_mario_bros
import argparse
import torch
import signal

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

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

def get_device():
    # this function returns the device to use for the model
    # check if CUDA is available and print some information
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available on device {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU instead.")
    return device

def parse_args():
    # this function parses the command-line arguments and returns the arguments object
    # create an argument parser object
    parser = argparse.ArgumentParser(description="Train a PPO model on Super Mario Bros.")
    parser.add_argument("model", type=str, help="the name of the model to use or 'new' for a default model")
    parser.add_argument("-n", "--num_cpu", type=int, default=2, help="the number of CPUs to use (default: 2)")
    parser.add_argument("-s", "--skip", type=int, default=2, help="the number of frames to skip (default: 2)")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.000025, help="the learning rate for the model (default: 0.000025)")
    parser.add_argument("-d", "--log_dir", type=str, default="tmp/", help="the log directory for the model (default: tmp/)")
    parser.add_argument("-e", "--env_id", type=str, default="SuperMarioBros-1-1-v0", help="the environment id for the game (default: SuperMarioBros-1-1-v0)")
    return parser.parse_args()

if __name__ == "__main__":
    
    # parse the command-line arguments
    args = parse_args()

    # print the hyperparameters
    print(f"Hyperparameters:")
    print(f"Number of CPUs: {args.num_cpu}")
    print(f"Skip: {args.skip}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Log directory: {args.log_dir}")
    print(f"Environment id: {args.env_id}")

    # get the device to use for the model
    device = get_device()

    # load the model from the file or create a new one
    if os.path.isfile(args.model):
        # load the model from the file
        env = VecMonitor(SubprocVecEnv([make_env(args.env_id, i, skip=args.skip) for i in range(args.num_cpu)]), args.log_dir + 'monitor')
        model = PPO.load(args.model, env=env, device=device)
        print("Model loaded from file.")

    elif args.model == "new":
        # create a default model with some parameters
        env = VecMonitor(SubprocVecEnv([make_env(args.env_id, i, skip=args.skip) for i in range(args.num_cpu)]), args.log_dir + 'monitor')
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_super_mario_bros_tensorboard/", learning_rate=args.learning_rate, device=device)
        print("New model created.")
    else:
        # raise an error if the model name is invalid
        raise ValueError(f"Invalid model name: {args.model}")

    train_model(model, args.log_dir, args.learning_rate)
