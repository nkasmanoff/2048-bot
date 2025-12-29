#!/usr/bin/env python3
"""Train RL agents on 2048.

Usage:
    # Train with REINFORCE
    python train.py --algorithm reinforce --episodes 50

    # Train with A2C (recommended)
    python train.py --algorithm a2c --episodes 50

    # Train with custom settings
    python train.py --algorithm a2c --episodes 100 --lr 0.001 --n-steps 32

    # Train from pretrained checkpoint
    python train.py --algorithm a2c --pretrained models/pretrained.pt

    # Record training videos
    python train.py --algorithm a2c --episodes 10 --record-video
"""

import argparse
import os

from src.agents import A2CAgent, REINFORCEAgent
from src.environment import train_agent


def main():
    parser = argparse.ArgumentParser(description="Train RL agents on 2048")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="a2c",
        choices=["reinforce", "a2c"],
        help="RL algorithm to use (default: a2c)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of training episodes (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden layer size (default: 128)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=32,
        help="N-step return size for A2C (default: 32)",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient for exploration (default: 0.01)",
    )
    parser.add_argument(
        "--action-delay",
        type=float,
        default=0.15,
        help="Delay between actions in seconds (default: 0.15)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Maximum steps per episode (default: 2000)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model to continue training",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record training videos",
    )
    args = parser.parse_args()

    # Create agent
    if args.algorithm == "reinforce":
        agent = REINFORCEAgent(
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
        )
        print("Training with REINFORCE algorithm")
    elif args.algorithm == "a2c":
        agent = A2CAgent(
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            n_steps=args.n_steps,
        )
        print("Training with A2C algorithm")
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    # Load pretrained weights if provided
    if args.pretrained and os.path.exists(args.pretrained):
        agent.load_model(args.pretrained)
        print(f"Loaded pretrained model from: {args.pretrained}")

    print(f"\nTraining Configuration:")
    print(f"  Algorithm: {args.algorithm.upper()}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Hidden Dim: {args.hidden_dim}")
    if args.algorithm == "a2c":
        print(f"  N-Steps: {args.n_steps}")
    print(f"  Entropy Coef: {args.entropy_coef}")
    print(f"  Action Delay: {args.action_delay}s")
    print(f"  Max Steps/Episode: {args.max_steps}")
    print()

    # Train
    metrics = train_agent(
        agent=agent,
        num_episodes=args.episodes,
        record_video=args.record_video,
        action_delay=args.action_delay,
        max_steps_per_episode=args.max_steps,
    )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Episodes: {args.episodes}")
    print(f"Best Score: {metrics['best_score']}")
    print(f"Best Max Tile: {metrics['best_max_tile']}")
    print(
        f"Final Avg Score (last 10): {sum(metrics['episode_scores'][-10:]) / min(10, len(metrics['episode_scores'])):.1f}"
    )


if __name__ == "__main__":
    main()
