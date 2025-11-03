import argparse, os, numpy as np, torch, gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from rl.agents.ppo import PPOAgent, PPOConfig
from rl.common.utils import set_seed

def train(env_id="CartPole-v1", total_steps=200_000, seed=42, device="cpu", checkpoint="runs/ppo_cartpole.pt",
          logdir=None):
    set_seed(seed)
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = PPOAgent(obs_dim, act_dim, PPOConfig(device=device))
    os.makedirs("runs", exist_ok=True)
    writer = SummaryWriter(logdir) if logdir else None

    steps_per_iter = 2048
    best= -1e9
    for it in trange(total_steps // steps_per_iter, desc="PPO"):
        traj = agent.collect_rollout(env, steps_per_iter)
        agent.update(traj)
        ep_ret = evaluate_once(env, agent)
        if writer:
            writer.add_scalar("ppo/return", ep_ret, it)
        if ep_ret > best:
            best = ep_ret
            agent.save(checkpoint)
    if writer: writer.close()
    print(f"Best eval return: {best:.1f}; saved to {checkpoint}")

def evaluate_once(env, agent, seed=123):
    obs, _ = env.reset(seed=seed)
    done=False; ep_ret=0.0
    while not done:
        x = torch.as_tensor(obs, dtype=torch.float32, device=agent.cfg.device).unsqueeze(0)
        logits, _ = agent.net(x)
        a = int(torch.argmax(logits, dim=-1).item())
        obs, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        ep_ret += r
    return ep_ret

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", type=str, default="CartPole-v1")
    ap.add_argument("--total-steps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--checkpoint", type=str, default="runs/ppo_cartpole.pt")
    ap.add_argument("--logdir", type=str, default=None)
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    if args.eval:
        env = gym.make(args.env_id)
        from rl.agents.ppo import PPOAgent, PPOConfig
        agent = PPOAgent(env.observation_space.shape[0], env.action_space.n, PPOConfig(device=args.device))
        agent.load(args.checkpoint)
        print("Eval one episode return:", evaluate_once(env, agent))
    else:
        train(env_id=args.env_id, total_steps=args.total_steps, seed=args.seed, device=args.device,
              checkpoint=args.checkpoint, logdir=args.logdir)

if __name__ == "__main__":
    main()

