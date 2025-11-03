try:
    import gymnasium as gym
    from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
except Exception:
    gym = None

def make_env(env_id: str, frame_stack: int = 4, grayscale: bool = True, resize: int = 84):
    if gym is None:
        raise ImportError("gymnasium not installed.")
    env = gym.make(env_id)
    if len(env.observation_space.shape) >= 2:
        if grayscale:
            env = GrayScaleObservation(env, keep_dim=True)
        if resize is not None:
            env = ResizeObservation(env, resize)
        if frame_stack and frame_stack > 1:
            env = FrameStack(env, num_stack=frame_stack)
    return env

def is_image_observation(env):
    if gym is None:
        return False
    return len(env.observation_space.shape) >= 2

