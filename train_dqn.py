import argparse, os, numpy as np, torch, gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from rl.agents.dqn import DQNAgent, DQNConfig, QNet
from rl.agents.dqn_cnn import QNetCNN
from rl.common.utils import set_seed
from rl.common.wrappers import make_env, is_image_observation

def train(env_id="CartPole-v1", episodes=400, seed=42, device="cpu", checkpoint="runs/dqn_cartpole.pt",
          logdir=None, cnn=False, frame_stack=4):
    set_seed(seed)
    try:
        env = make_env(env_id, frame_stack=frame_stack) if cnn else gym.make(env_id)
    except Exception:
        env = gym.make(env_id)

    obs_space = env.observation_space
    act_dim = env.action_space.n
    if cnn or is_image_observation(env):
        in_ch = obs_space.shape[-1] if len(obs_space.shape)==3 else 1
        net_ctor = lambda obs_dim, act_dim: QNetCNN(in_ch, act_dim)
        obs_dim = None
    else:
        net_ctor = lambda obs_dim, act_dim: QNet(obs_dim, act_dim)
        obs_dim = obs_space.shape[0]

    cfg = DQNConfig(device=device)
    agent = DQNAgent(obs_dim or 4, act_dim, cfg)
    if cnn or is_image_observation(env):
        agent.q = net_ctor(None, act_dim).to(device)
        agent.q_target = net_ctor(None, act_dim).to(device)
        agent.q_target.load_state_dict(agent.q.state_dict())

    os.makedirs("runs", exist_ok=True)
    writer = SummaryWriter(logdir) if logdir else None
    best = -1e9

    for ep in trange(episodes, desc="DQN"):
        obs, _ = env.reset(seed=seed+ep)
        ep_ret = 0.0
        done = False
        while not done:
            if cnn or is_image_observation(env):
                s = torch.as_tensor(np.transpose(obs, (2,0,1)), dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q = agent.q(s)
                a = int(q.argmax(dim=-1).item())
            else:
                a = agent.act(obs)
            next_obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            agent.push(obs, a, r, next_obs, float(done))
            obs = next_obs
            ep_ret += r
            loss = agent.train_step()

        if writer:
            writer.add_scalar("dqn/return", ep_ret, ep)
            if loss is not None:
                writer.add_scalar("dqn/loss", loss, ep)
            writer.add_scalar("dqn/epsilon", agent.epsilon(), ep)

        if ep_ret > best:
            best = ep_ret
            agent.save(checkpoint)
        if (ep+1) % 20 == 0:
            print(f"Episode {ep+1}: return={ep_ret:.1f} epsilon={agent.epsilon():.3f}")

    if writer: writer.close()
    print(f"Best return: {best:.1f}; saved to {checkpoint}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", type=str, default="CartPole-v1")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--checkpoint", type=str, default="runs/dqn_cartpole.pt")
    ap.add_argument("--logdir", type=str, default=None)
    ap.add_argument("--cnn", action="store_true")
    ap.add_argument("--frame-stack", type=int, default=4)
    args = ap.parse_args()
    train(env_id=args.env_id, episodes=args.episodes, seed=args.seed, device=args.device,
          checkpoint=args.checkpoint, logdir=args.logdir, cnn=args.cnn, frame_stack=args.frame_stack)

if __name__ == "__main__":
    main()

