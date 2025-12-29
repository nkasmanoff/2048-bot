#!/usr/bin/env python3
"""Example script demonstrating LLM agent usage for 2048.

This script shows how to:
1. Create an LLM agent with different models
2. Run games with the LLM agent
3. Access and analyze the interaction logs

Requirements:
- Set OPENAI_API_KEY environment variable
- Install required packages: pip install -r requirements.txt
"""


import os
from dotenv import load_dotenv

from src.llm_agent import create_llm_agent
from src.environment import load_and_play

# Load environment variables from .env file
load_dotenv()


def main():
    print("=" * 60)
    print("LLM Agent Example for 2048")
    print("=" * 60)

    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ Error: API key not found!")
        print("\nPlease create a .env file with:")
        print("  OPENROUTER_API_KEY=your-api-key-here")
        print("  LLM_MODEL=allenai/olmo-3-7b-instruct")
        print("  LLM_BASE_URL=https://openrouter.ai/api/v1")
        print("\nOr set environment variables directly:")
        print("  export OPENROUTER_API_KEY='your-api-key-here'")
        return

    # Create LLM agent (reads config from .env)
    print("\nCreating LLM agent from .env configuration...")
    agent = create_llm_agent(verbose=True)

    # Play games
    num_games = 3
    print(f"\n{'='*60}")
    print(f"Playing {num_games} games with LLM agent")
    print(f"{'='*60}\n")

    metrics = load_and_play(
        agent=agent,
        model_path=None,  # LLM agent doesn't need a model file
        num_games=num_games,
        record_video=False,
        action_delay=0.2,  # Slower to see LLM's reasoning
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Games played: {num_games}")
    print(f"Best Score: {metrics['best_score']}")
    print(f"Best Max Tile: {metrics['best_max_tile']}")
    print(f"Average Score: {metrics['avg_score']:.1f}")
    print(f"Average Max Tile: {metrics['avg_max_tile']:.1f}")
    print(f"\n📝 Detailed logs saved to: {agent.log_file}")
    print("\nYou can review the LLM's thinking process in the log file!")
    print("=" * 60)


if __name__ == "__main__":
    main()
