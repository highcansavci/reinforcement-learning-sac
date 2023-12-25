import gym
import numpy as np
from agent.sac_agent import Agent


if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    env = gym.wrappers.RecordVideo(env, 'video')
    num_horizon = 20
    batch_size = 5
    n_epochs = 4
    alpha = 3e-4
    agent_ = Agent(env=env, input_dims=env.observation_space.shape)
    agent_.load_models()
    n_games = 5

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent_.choose_action(observation)
            observation_, reward, done, info = env.step(np.squeeze(action))
            env.render()
            score += reward
            observation = observation_