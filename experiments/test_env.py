import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("Pendulum-v1", render_mode="human", g=9.81)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

state, info = env.reset()
done = False
rewards = []

while not done:
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    done = terminated or truncated
    env.render()

plt.plot(rewards)
plt.show()