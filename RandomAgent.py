import gym_super_mario_bros

env = gym_super_mario_bros.make('SuperMarioBros-v0')
obs = env.reset()

print(obs.shape)

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
