"""Common utilities for the 2048 bot.

Contains shared helper functions for browser setup, game interaction,
and observation extraction used across multiple modules.
"""

import time

import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Observation dimensions:
# 16 tile values (4x4 grid, log2 normalized) +
# 4 merge availability indicators (one per direction) +
# 1 empty cell count (normalized) +
# 1 max tile (log2 normalized) +
# 1 current score (log normalized)
OBSERVATION_DIM = 23

# Discrete action space: 4 directions (Up, Right, Down, Left)
NUM_ACTIONS = 4

# Action mapping
ACTION_UP = 0
ACTION_RIGHT = 1
ACTION_DOWN = 2
ACTION_LEFT = 3

ACTION_NAMES = ["Up", "Right", "Down", "Left"]
ACTION_KEYS = ["ArrowUp", "ArrowRight", "ArrowDown", "ArrowLeft"]


def setup_browser(headless: bool = False) -> webdriver.Chrome:
    """Set up and return a Chrome WebDriver instance.

    Args:
        headless: Whether to run Chrome in headless mode.

    Returns:
        Configured Chrome WebDriver instance.
    """
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(options=options)
    driver.get("https://2048.io/")
    print("Waiting for game to load...")
    time.sleep(3)

    return driver


def start_game(driver: webdriver.Chrome) -> None:
    """Start a new game by clicking the New Game button or pressing a key.

    Args:
        driver: Chrome WebDriver instance.
    """
    try:
        # Try to find and click the "New Game" button
        new_game_buttons = driver.find_elements(By.CLASS_NAME, "restart-button")
        for button in new_game_buttons:
            if button.is_displayed():
                button.click()
                print("Clicked New Game button")
                time.sleep(1)
                return

        # Alternative: look for button with "New Game" text
        buttons = driver.find_elements(By.TAG_NAME, "a")
        for button in buttons:
            if (
                "New Game" in button.text
                or "new-game" in button.get_attribute("class")
                or ""
            ):
                button.click()
                print("Clicked New Game link")
                time.sleep(1)
                return

    except Exception as e:
        print(f"Could not find New Game button: {e}")

    print("Game ready!")


def wait_for_game_ready(controller, max_wait: float = 10.0) -> bool:
    """Wait for the game to be ready (board loaded).

    Args:
        controller: Game2048Controller instance.
        max_wait: Maximum time to wait in seconds.

    Returns:
        True if game is ready, False otherwise.
    """
    start_time = time.time()
    while time.time() - start_time < max_wait:
        state = controller.get_game_state()
        if state and state.get("board") is not None:
            return True
        time.sleep(0.5)
    return False


def extract_observation(state: dict | None) -> np.ndarray:
    """Extract observation vector from game state.

    Args:
        state: Game state from Game2048Controller.get_game_state().

    Returns:
        23-dimensional observation vector (dtype float32).

    Observation features:
        0-15: Board tiles (4x4 grid), log2 normalized to [0, 1]
              where 0 = empty, 2048 = 11, so divide by 11
        16-19: Move availability (up, right, down, left) - binary
        20: Empty cell count (normalized by 16)
        21: Max tile value (log2 normalized)
        22: Current score (log10 normalized)
    """
    if not state or state.get("board") is None:
        return np.zeros(OBSERVATION_DIM, dtype=np.float32)

    board = state.get("board", [[0] * 4 for _ in range(4)])
    score = state.get("score", 0)

    # Flatten and normalize board (log2 scale)
    # 2^11 = 2048, so normalize by 11 for the main goal
    # but tiles can go higher (4096, 8192, etc.)
    board_flat = []
    max_tile = 0
    empty_count = 0

    for row in board:
        for tile in row:
            if tile == 0:
                board_flat.append(0.0)
                empty_count += 1
            else:
                # log2(tile) normalized, e.g., log2(2048) = 11
                log_val = np.log2(tile) / 15.0  # Normalize by 15 to handle higher tiles
                board_flat.append(min(log_val, 1.0))
                max_tile = max(max_tile, tile)

    # Check move availability in each direction
    can_move_up = _can_move(board, ACTION_UP)
    can_move_right = _can_move(board, ACTION_RIGHT)
    can_move_down = _can_move(board, ACTION_DOWN)
    can_move_left = _can_move(board, ACTION_LEFT)

    # Max tile normalized
    max_tile_norm = np.log2(max_tile) / 15.0 if max_tile > 0 else 0.0

    # Score normalized (log10 scale)
    score_norm = np.log10(score + 1) / 6.0 if score >= 0 else 0.0  # ~6 for 1M score

    # Empty cells normalized
    empty_norm = empty_count / 16.0

    observation = np.array(
        board_flat  # 16 values
        + [
            float(can_move_up),
            float(can_move_right),
            float(can_move_down),
            float(can_move_left),
            empty_norm,
            max_tile_norm,
            score_norm,
        ],
        dtype=np.float32,
    )

    return observation


def _can_move(board: list, direction: int) -> bool:
    """Check if a move in the given direction is valid.

    Args:
        board: 4x4 list of tile values.
        direction: Action index (0=Up, 1=Right, 2=Down, 3=Left).

    Returns:
        True if the move would change the board.
    """
    # Transpose for vertical moves
    if direction in [ACTION_UP, ACTION_DOWN]:
        board = list(zip(*board))

    # Reverse for right/down moves
    reverse = direction in [ACTION_RIGHT, ACTION_DOWN]

    for row in board:
        row = list(row)
        if reverse:
            row = row[::-1]

        # Check if tiles can slide or merge
        non_zero = [x for x in row if x != 0]

        # Check if any zeros are between non-zero values (can slide)
        if len(non_zero) < len([x for x in row if x != 0]) or len(non_zero) != len(
            row
        ) - row.count(0):
            # Can slide if there are gaps
            original_positions = [i for i, x in enumerate(row) if x != 0]
            if original_positions and original_positions[0] != 0:
                return True
            for i in range(1, len(original_positions)):
                if original_positions[i] != original_positions[i - 1] + 1:
                    return True

        # Check for possible merges
        for i in range(len(non_zero) - 1):
            if non_zero[i] == non_zero[i + 1]:
                return True

    return False


def action_to_key(action: int) -> str:
    """Convert action index to key name.

    Args:
        action: Action index (0-3).

    Returns:
        Key name string.
    """
    return ACTION_KEYS[action]


def action_to_name(action: int) -> str:
    """Convert action index to human-readable name.

    Args:
        action: Action index (0-3).

    Returns:
        Action name string.
    """
    return ACTION_NAMES[action]
