import gymnasium as gym
from rl.agents.dqn import DQNAgent, DQNConfig

def test_dqn_one_step():
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = DQNAgent(obs_dim, act_dim, DQNConfig(device="cpu"))
    obs, _ = env.reset(seed=0)
    a = agent.act(obs)
    assert 0 <= a < act_dim

