# 2048 Bot

An autonomous bot that plays 2048 using Selenium WebDriver, with both random baseline and reinforcement learning approaches.

## Features

-   **Random Agent**: Baseline agent with uniform random action selection
-   **RL Agents**: REINFORCE and A2C (Actor-Critic) agents that learn through online training
-   **LLM Agent**: Uses language models (GPT-4, GPT-3.5, etc.) to play via reasoning and strategic thinking
-   **Video Recording**: Record gameplay videos with state annotations
-   **State Extraction**: Extracts game board data via JavaScript injection

## Project Structure

```
2048-bot/
├── src/                    # Core library modules
│   ├── __init__.py         # Package exports
│   ├── agents.py           # RL agents (Random, REINFORCE, A2C)
│   ├── llm_agent.py        # LLM-based agent using OpenAI API
│   ├── controller.py       # Selenium game controller
│   ├── environment.py      # Gym environment wrapper
│   └── utils.py            # Shared helper functions
├── train.py                # Train RL agents
├── play.py                 # Play with trained models
├── models/                 # Saved model checkpoints
├── training_logs/          # Training metrics
├── inference_logs/         # Inference metrics
├── llm_logs/               # LLM agent interaction logs
└── games/                  # Recorded gameplay videos
```

## Installation

```bash
pip install -r requirements.txt
```

You'll also need ChromeDriver installed and in your PATH.

For LLM agent, create a `.env` file:

```bash
cp env.example .env
# Edit .env with your API key and model preferences
```

## Quick Start

### 1. Play with Random Agent (Baseline)

```bash
python play.py --agent random --games 5
```

### 2. Train an RL Agent

```bash
# Train with A2C (recommended)
python train.py --algorithm a2c --episodes 50

# Train with REINFORCE
python train.py --algorithm reinforce --episodes 50
```

### 3. Play with a Trained Model

```bash
python play.py --agent a2c --model models/best_model_2048.pt --games 5
```

### 4. Play with LLM Agent

```bash
# Create a .env file with your configuration
cp env.example .env
# Edit .env and add your API key and model preferences

# Play with LLM agent (reads config from .env)
python play.py --agent llm --games 3
```

## Usage Examples

### Training

```bash
# Basic A2C training
python train.py --algorithm a2c --episodes 100

# A2C with custom N-step updates
python train.py --algorithm a2c --episodes 100 --n-steps 32

# Train from pre-trained checkpoint
python train.py --algorithm a2c --pretrained models/pretrained.pt

# Record training videos
python train.py --algorithm a2c --episodes 10 --record-video

# Faster/slower decision making
python train.py --action-delay 0.1   # Faster (10 Hz)
python train.py --action-delay 0.2   # Slower, more stable (5 Hz)
```

### Playing / Inference

```bash
# Play with A2C model
python play.py --agent a2c --model models/best_model_2048.pt

# Play with REINFORCE model
python play.py --agent reinforce --model models/best_model_2048.pt

# Play with random agent (baseline)
python play.py --agent random

# Play with LLM agent (requires .env file with configuration)
python play.py --agent llm --games 3

# Record gameplay
python play.py --agent a2c --model models/best_model_2048.pt --record-video
```

### LLM Agent Configuration

The LLM agent uses a `.env` file for configuration. Create it from the example:

```bash
cp env.example .env
```

Edit `.env` with your settings:

```bash
# Required: API key
OPENROUTER_API_KEY=your-key-here
# or
OPENAI_API_KEY=your-key-here

# Optional: Model name (defaults to gpt-4o-mini)
LLM_MODEL=allenai/olmo-3-7b-instruct

# Optional: Base URL (required for OpenRouter)
LLM_BASE_URL=https://openrouter.ai/api/v1
```

**Supported Providers:**

-   **OpenAI**: Use `OPENAI_API_KEY` and omit `LLM_BASE_URL`
-   **OpenRouter**: Use `OPENROUTER_API_KEY` and set `LLM_BASE_URL=https://openrouter.ai/api/v1`
-   **Other OpenAI-compatible APIs**: Set appropriate `LLM_BASE_URL`

**Model Examples:**

```bash
# OpenAI models
LLM_MODEL=gpt-4o
LLM_MODEL=gpt-4o-mini
LLM_MODEL=gpt-3.5-turbo

# OpenRouter models (many free and open source options)
LLM_MODEL=allenai/olmo-3-7b-instruct
LLM_MODEL=meta-llama/llama-3.2-3b-instruct:free
LLM_MODEL=anthropic/claude-3.5-sonnet
LLM_MODEL=google/gemini-2.0-flash-thinking-exp:free
```

## How It Works

### Game Controller

The bot uses Selenium WebDriver to interact with [2048.io](https://2048.io/). It injects JavaScript to read game state (board tiles, score, game over status) and controls the game by sending arrow key presses.

### Observation Space (23 dimensions)

| Features | Description                               |
| -------- | ----------------------------------------- |
| 0-15     | Board tiles (4×4 grid), log2 normalized   |
| 16-19    | Move availability (up, right, down, left) |
| 20       | Empty cell count (normalized)             |
| 21       | Max tile value (log2 normalized)          |
| 22       | Current score (log10 normalized)          |

### Action Space

Discrete with 4 actions:

-   0: Up
-   1: Right
-   2: Down
-   3: Left

### Reward Function

| Event                  | Reward                            |
| ---------------------- | --------------------------------- |
| Score increase         | `+log2(score_increase + 1) × 0.1` |
| Invalid move           | `-0.5`                            |
| New max tile milestone | `+log2(max_tile) × 0.5`           |
| Survival (per step)    | `+0.01`                           |
| Game over              | `-2.0 + log2(max_tile) × 0.1`     |
| Win (reach 2048)       | `+10.0`                           |

### Agents

-   **REINFORCE**: Updates policy at end of episode using discounted returns. Higher variance but simpler.
-   **A2C**: N-step updates during episode with separate actor/critic networks. More stable for long games.
-   **LLM Agent**: Uses large language models (via OpenAI-compatible APIs) to reason about game state and make strategic decisions. The LLM receives the board state as text and outputs thinking (in `<thinking>` tags) followed by an action (in `<action>` tags). Configuration is managed through `.env` file. All interactions are logged to `llm_logs/` for analysis. Supports OpenAI, OpenRouter, and other compatible APIs.

## Output Files

### Training Logs (`training_logs/`)

```json
{
  "episode_scores": [120, 256, 512, ...],
  "episode_max_tiles": [64, 128, 256, ...],
  "best_score": 15000,
  "best_max_tile": 1024,
  "timestamp": "20251222_123456"
}
```

### Inference Logs (`inference_logs/`)

```json
{
    "game_scores": [1200, 2400, 3600],
    "game_max_tiles": [256, 512, 512],
    "best_score": 3600,
    "avg_score": 2400.0
}
```

### LLM Logs (`llm_logs/`)

Text logs showing LLM reasoning and actions:

```
[2025-12-23 10:30:45] Step 15
[2025-12-23 10:30:45] 💭 LLM Thinking:
   The board has high tiles in the top-left corner. I should move left
   to keep them together and create merge opportunities...
[2025-12-23 10:30:45] 🎮 Action Selected: Left (index: 3)
[2025-12-23 10:30:45] 💰 Reward: 0.85
```

## Tips

-   **A2C is recommended** over REINFORCE for 2048's long episodes
-   Use `--action-delay 0.1` for faster gameplay, `0.15-0.2` for more stable training
-   The game requires merging tiles strategically — keeping the highest tile in a corner is a common human strategy
-   Record videos (`--record-video`) to debug agent behavior
-   Check `training_logs/` and `inference_logs/` for metrics

## Requirements

-   Python 3.10+
-   Chrome browser
-   ChromeDriver (matching your Chrome version)
-   OpenAI API key (for LLM agent only)
-   See `requirements.txt` for Python packages:
    -   gymnasium
    -   numpy
    -   selenium
    -   opencv-python
    -   Pillow
    -   torch
    -   openai (for LLM agent)
