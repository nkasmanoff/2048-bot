#!/usr/bin/env python3
"""Train RL agents on 2048.

Usage:
    # Train with PPO (recommended, default)
    python train.py --algorithm ppo --episodes 50

    # Train with Weights & Biases logging
    python train.py --algorithm ppo --episodes 50 --wandb

    # Train with custom wandb project/run name
    python train.py --algorithm ppo --episodes 50 --wandb --wandb-project my-2048 --wandb-run experiment-1

    # Train with A2C
    python train.py --algorithm a2c --episodes 50

    # Train with REINFORCE
    python train.py --algorithm reinforce --episodes 50

    # Train with custom settings
    python train.py --algorithm ppo --episodes 100 --lr 0.0003 --n-steps 128

    # Train from pretrained checkpoint
    python train.py --algorithm ppo --pretrained models/pretrained.pt

    # Record training videos
    python train.py --algorithm ppo --episodes 10 --record-video
"""

import argparse
import os

from src.agents import A2CAgent, PPOAgent, REINFORCEAgent
from src.environment import train_agent


def main():
    parser = argparse.ArgumentParser(description="Train RL agents on 2048")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["reinforce", "a2c", "ppo"],
        help="RL algorithm to use (default: ppo)",
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
        default=128,
        help="N-step rollout size for PPO/A2C (default: 128)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=4,
        help="Number of epochs per PPO update (default: 4)",
    )
    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=0.2,
        help="PPO clipping parameter (default: 0.2)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda for PPO (default: 0.95)",
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
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="2048-bot",
        help="Weights & Biases project name (default: 2048-bot)",
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        default=None,
        help="Weights & Biases run name (default: auto-generated)",
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
    elif args.algorithm == "ppo":
        agent = PPOAgent(
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            clip_epsilon=args.clip_epsilon,
            gae_lambda=args.gae_lambda,
        )
        print("Training with PPO algorithm")
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
    if args.algorithm in ("a2c", "ppo"):
        print(f"  N-Steps: {args.n_steps}")
    if args.algorithm == "ppo":
        print(f"  N-Epochs: {args.n_epochs}")
        print(f"  Clip Epsilon: {args.clip_epsilon}")
        print(f"  GAE Lambda: {args.gae_lambda}")
    print(f"  Entropy Coef: {args.entropy_coef}")
    print(f"  Action Delay: {args.action_delay}s")
    print(f"  Max Steps/Episode: {args.max_steps}")
    print(f"  Wandb Logging: {'Enabled' if args.wandb else 'Disabled'}")
    print()

    # Initialize wandb if enabled
    if args.wandb:
        import wandb
        
        config = {
            "algorithm": args.algorithm,
            "episodes": args.episodes,
            "learning_rate": args.lr,
            "gamma": args.gamma,
            "hidden_dim": args.hidden_dim,
            "entropy_coef": args.entropy_coef,
            "action_delay": args.action_delay,
            "max_steps": args.max_steps,
        }
        if args.algorithm in ("a2c", "ppo"):
            config["n_steps"] = args.n_steps
        if args.algorithm == "ppo":
            config["n_epochs"] = args.n_epochs
            config["clip_epsilon"] = args.clip_epsilon
            config["gae_lambda"] = args.gae_lambda
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=config,
        )
        print(f"Wandb run initialized: {wandb.run.name}")
        print(f"Wandb project: {args.wandb_project}")
        print()

    # Train
    metrics = train_agent(
        agent=agent,
        num_episodes=args.episodes,
        record_video=args.record_video,
        action_delay=args.action_delay,
        max_steps_per_episode=args.max_steps,
        use_wandb=args.wandb,
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

    # Finish wandb run
    if args.wandb:
        import wandb
        
        # Log final summary metrics
        wandb.summary["best_score"] = metrics["best_score"]
        wandb.summary["best_max_tile"] = metrics["best_max_tile"]
        wandb.summary["final_avg_score"] = sum(metrics['episode_scores'][-10:]) / min(10, len(metrics['episode_scores']))
        wandb.finish()
        print("\nWandb run finished and synced.")


if __name__ == "__main__":
    main()
