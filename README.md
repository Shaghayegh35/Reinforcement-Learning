# 🧠 Reinforcement Learning Pro — DQN & PPO with TensorBoard, Atari, and CI

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-green?logo=openai)](https://gymnasium.farama.org/)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-visualization-orange?logo=tensorflow)](https://www.tensorflow.org/tensorboard)
[![Tests](https://github.com/yourusername/Reinforcement-Learning-Pro/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/Reinforcement-Learning-Pro/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](LICENSE)

A **modern, production-ready reinforcement learning framework** implementing **Deep Q-Networks (DQN)** and **Proximal Policy Optimization (PPO)** with full support for **TensorBoard visualization**, **Atari environments**, and **automated testing + CI/CD**.

---

## 🚀 Key Features
- 🧠 **DQN** with experience replay, target networks, and CNN backbone for image-based environments  
- 🌀 **PPO** with clipped objective, entropy regularization, and GAE advantage estimation  
- 📊 **TensorBoard integration** for loss, returns, and epsilon decay tracking  
- 🕹️ **Atari-ready wrappers** with grayscale, resize, and frame stack utilities  
- 🧪 **Unit tests** (`pytest`) for reproducibility and fast validation  
- ⚙️ **GitHub Actions CI** workflow for continuous testing  

---

## ⚡ Quickstart

```bash
# Create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements_additions.txt

# Train agents
python train_dqn.py --episodes 400 --logdir runs/tb/dqn_cartpole
python train_ppo.py --total-steps 200000 --logdir runs/tb/ppo_cartpole

# Visualize
tensorboard --logdir runs/tb
