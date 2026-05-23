import torch
import matplotlib.pyplot as plt
import numpy as np
from reinforce.policy_network import PolicyNetwork
import gymnasium as gym

env = gym.make("Pendulum-v1", render_mode="human", g=9.81)

state, info = env.reset()
done = False
rewards = []

# use randomised weights policy
policy_network = PolicyNetwork()

# make state a tensor
state = torch.tensor(state, dtype=torch.float32)
while not done:

    # sample action given state from Gaussian
    mean, log_std = policy_network(state)
    std = torch.exp(log_std)

    sample_action =  torch.normal(mean, std).detach().numpy()

    next_state, reward, terminated, truncated, info = env.step(sample_action)
    state = torch.tensor(next_state, dtype=torch.float32)
    rewards.append(reward)
    done = terminated or truncated
    env.render()

plt.plot(rewards)
plt.show()
