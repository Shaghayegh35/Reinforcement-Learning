import pytest
try:
    from rl.common.wrappers import make_env, is_image_observation
    import gymnasium as gym
except Exception:
    gym = None

@pytest.mark.skipif(gym is None, reason="gymnasium not installed in CI minimal env")
def test_make_env_cartpole():
    env = make_env("CartPole-v1", frame_stack=1)
    assert env is not None

