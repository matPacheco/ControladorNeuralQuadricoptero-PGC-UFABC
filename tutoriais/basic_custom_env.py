import gymnasium as gym


env = gym.make("GPS-distance-v0", render_mode="human")
print("Action Space:")
print(env.action_space)  # Discrete(4) <class 'gymnasium.spaces.discrete.Discrete'>
print("Observation Space:")
print(env.observation_space)
observation, info = env.reset(seed=42)

for _ in range(10):
   action = env.action_space.sample()  # this is where you would insert your policy
   print("Action", action)
   observation, reward, terminated, truncated, info = env.step(action)

   print("Observation:", observation)
   print("Reward:", reward)
   print("Terminated:", terminated)
   print("Truncated:", truncated)
   print("Info:", info)

   print("------------")

   if terminated or truncated:
      observation, info = env.reset()

env.close()