from stable_baselines3 import PPO 
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import gym_super_mario_bros

# Load the model with custom_objects
model = PPO.load("tmp/best_model.zip")

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = MaxAndSkipEnv(env, skip=1)

    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs.copy())
        obs, rewards, done, info = env.step(action)
        env.render()

if __name__ == "__main__":
    main()