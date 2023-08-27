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

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym_super_mario_bros.make(env_id)
        env = MaxAndSkipEnv(env, skip=4)
        env.seed(seed + rank)
        # env = VecMonitor(env)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    env_id = "SuperMarioBros-v0"
    num_cpu = 2

    # check env is compatible with VecMonitor and SubprocVecEnv

    env = VecMonitor(SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)]), 'tmp/monitor')

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_super_mario_bros_tensorboard/", learning_rate=0.00003)

    print("Training...")
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=10000000, callback=callback, tb_log_name="ppo_test00003")
    model.save("ppo_super_mario_bros")
    print("Done!")

