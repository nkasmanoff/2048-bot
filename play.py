#!/usr/bin/env python3
"""Play 2048 with a trained agent or random baseline.

Usage:
    # Play with random agent (baseline)
    python play.py --agent random --games 5

    # Play with REINFORCE model
    python play.py --agent reinforce --model models/best_model_2048.pt --games 5

    # Play with A2C model
    python play.py --agent a2c --model models/best_model_2048.pt --games 5

    # Play with LLM agent (requires .env file with API key and model config)
    python play.py --agent llm --games 3

    # Record gameplay video
    python play.py --agent random --games 3 --record-video
"""

import argparse

from dotenv import load_dotenv

from src.agents import A2CAgent, RandomAgent, REINFORCEAgent
from src.llm_agent import create_llm_agent
from src.environment import load_and_play

# Load environment variables from .env file
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Play 2048 with trained agents")
    parser.add_argument(
        "--agent",
        type=str,
        default="random",
        choices=["random", "reinforce", "a2c", "llm"],
        help="Agent type to use (default: random)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model weights (required for RL agents)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=5,
        help="Number of games to play (default: 5)",
    )
    parser.add_argument(
        "--action-delay",
        type=float,
        default=0.1,
        help="Delay between actions in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record gameplay video",
    )
    args = parser.parse_args()

    # Create agent
    if args.agent == "random":
        agent = RandomAgent()
        print("Using Random Agent (baseline)")
    elif args.agent == "reinforce":
        agent = REINFORCEAgent()
        print("Using REINFORCE Agent")
    elif args.agent == "a2c":
        agent = A2CAgent()
        print("Using A2C Agent")
    elif args.agent == "llm":
        agent = create_llm_agent(verbose=True)
        print("Using LLM Agent")
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")

    # Load and play
    metrics = load_and_play(
        agent=agent,
        model_path=args.model,
        num_games=args.games,
        record_video=args.record_video,
        action_delay=args.action_delay,
    )

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"Agent: {args.agent}")
    print(f"Games played: {args.games}")
    print(f"Best Score: {metrics['best_score']}")
    print(f"Best Max Tile: {metrics['best_max_tile']}")
    print(f"Average Score: {metrics['avg_score']:.1f}")
    print(f"Average Max Tile: {metrics['avg_max_tile']:.1f}")


if __name__ == "__main__":
    main()
