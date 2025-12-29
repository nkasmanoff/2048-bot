"""2048 Bot - RL training environment for 2048.io game.

This package provides:
- Game2048Env: Gymnasium environment wrapper
- Game2048Controller: Selenium-based game controller
- RL Agents: RandomAgent, REINFORCEAgent, A2CAgent
- Utility functions for browser setup and observation extraction
"""

from .agents import A2CAgent, RandomAgent, REINFORCEAgent
from .controller import Game2048Controller
from .environment import Game2048Env, load_and_play, setup_browser_and_game, train_agent
from .utils import (
    NUM_ACTIONS,
    OBSERVATION_DIM,
    extract_observation,
    setup_browser,
    start_game,
)

__all__ = [
    # Environment
    "Game2048Env",
    "Game2048Controller",
    # Agents
    "RandomAgent",
    "REINFORCEAgent",
    "A2CAgent",
    # Functions
    "setup_browser",
    "start_game",
    "setup_browser_and_game",
    "load_and_play",
    "train_agent",
    "extract_observation",
    # Constants
    "NUM_ACTIONS",
    "OBSERVATION_DIM",
]
