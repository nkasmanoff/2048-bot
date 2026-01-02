"""Gym environment wrapper for 2048 with RL training support.

Provides Game2048Env class and training functions for RL algorithms.
"""

import json
import os
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
from selenium.webdriver.common.by import By

from .controller import Game2048Controller
from .utils import (
    NUM_ACTIONS,
    OBSERVATION_DIM,
    extract_observation,
    setup_browser,
    start_game,
)

# Stagnation detection (no score increase)
STAGNATION_WINDOW_SIZE = 50
STAGNATION_THRESHOLD = 0


class Game2048Env(gym.Env):
    """Gym environment wrapper for 2048 game."""

    def __init__(
        self, driver, save_screenshots=False, record_video=False, action_delay=0.15
    ):
        """Initialize the 2048 environment.

        Args:
            driver: Selenium WebDriver instance.
            save_screenshots: Whether to save screenshots to disk.
            record_video: Whether to record gameplay video.
            action_delay: Delay between actions in seconds.
        """
        super().__init__()
        self.controller = Game2048Controller(driver, save_screenshots, record_video)
        self.action_delay = action_delay

        # Discrete action space: 4 directions (Up, Right, Down, Left)
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(OBSERVATION_DIM,), dtype=np.float32
        )

        self.previous_score = 0
        self.previous_max_tile = 0
        self.steps = 0
        self.invalid_moves = 0
        self.driver = driver
        self.last_action = None

    def _get_obs(self, state=None):
        """Extract observation from game state."""
        if state is None:
            state = self.controller.get_game_state()
        return extract_observation(state)

    def _get_max_tile(self, state):
        """Get the maximum tile from board state."""
        if not state or not state.get("board"):
            return 0
        max_tile = 0
        for row in state["board"]:
            for tile in row:
                max_tile = max(max_tile, tile)
        return max_tile

    def step(self, action, observation=None, probabilities=None):
        """Execute action and return next state, reward, done, info.

        Args:
            action: Discrete action index (0=Up, 1=Right, 2=Down, 3=Left).
            observation: Optional observation for annotation.
            probabilities: Optional action probabilities for annotation.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        # Execute the move
        move_result = self.controller.move(action)

        # Store action for tracking
        self.last_action = action

        if observation is not None or probabilities is not None:
            self.controller.set_frame_annotation(
                observation=observation, probabilities=probabilities, action=action
            )

        # Wait for animation
        time.sleep(self.action_delay)

        # Get new state
        full_state = self.controller.get_game_state()
        obs = self._get_obs(state=full_state)

        current_score = full_state.get("score", 0) if full_state else 0
        current_max_tile = self._get_max_tile(full_state)
        done = full_state.get("is_game_over", False) if full_state else False
        won = full_state.get("won", False) if full_state else False

        score_increase = current_score - self.previous_score

        # === Reward Shaping ===
        reward = 0.0

        # Reward for score increase (merging tiles)
        if score_increase > 0:
            # Log scale reward to balance early vs late game
            reward += np.log2(score_increase + 1) * 0.1

        # Penalty for invalid moves (move didn't change board)
        if not move_result.get("move_valid", True):
            reward -= 0.5
            self.invalid_moves += 1

        # Bonus for reaching new max tile milestones
        if current_max_tile > self.previous_max_tile:
            # Larger bonus for higher tiles
            tile_bonus = np.log2(current_max_tile) * 0.5 if current_max_tile > 0 else 0
            reward += tile_bonus

        # Small survival bonus
        reward += 0.01

        self.previous_score = current_score
        self.previous_max_tile = current_max_tile
        self.steps += 1

        if done:
            # Penalty for game over
            reward -= 2.0
            # But give credit for max tile achieved
            if current_max_tile > 0:
                reward += np.log2(current_max_tile) * 0.1

        if won:
            # Big bonus for winning (reaching 2048)
            reward += 10.0

        info = {
            "score": current_score,
            "max_tile": current_max_tile,
            "steps": self.steps,
            "score_increase": score_increase,
            "move_valid": move_result.get("move_valid", True),
            "invalid_moves": self.invalid_moves,
            "won": won,
        }

        return obs, reward, done, info

    def reset(self):
        """Reset environment for new episode."""
        # Reset the game
        self.controller.reset_game()
        time.sleep(0.5)

        # Wait for game to be ready
        max_wait = 5
        start_time = time.time()
        while time.time() - start_time < max_wait:
            state = self.controller.get_game_state()
            if state and state.get("board"):
                break
            time.sleep(0.2)

        # Reset tracking variables
        self.previous_score = 0
        self.previous_max_tile = 0
        self.steps = 0
        self.invalid_moves = 0
        self.last_action = None

        return self._get_obs()


def setup_browser_and_game(record_video=False, action_delay=0.15):
    """Set up browser and start a new game.

    Returns:
        driver: WebDriver instance.
        env: Game2048Env instance.
    """
    driver = setup_browser()
    start_game(driver)
    env = Game2048Env(
        driver,
        save_screenshots=False,
        record_video=record_video,
        action_delay=action_delay,
    )
    print(
        f"Decision frequency: ~{1/action_delay:.1f} Hz (action_delay={action_delay}s)"
    )
    return driver, env


def train_agent(
    agent,
    num_episodes=100,
    record_video=False,
    action_delay=0.15,
    max_steps_per_episode=2000,
    use_wandb=False,
):
    """Train an RL agent on 2048.

    Args:
        agent: RL agent with select_action, store_reward, and update_policy methods.
        num_episodes: Number of training episodes.
        record_video: Whether to record gameplay video.
        action_delay: Delay between actions.
        max_steps_per_episode: Maximum steps before ending episode.
        use_wandb: Whether to log metrics to Weights & Biases.

    Returns:
        Training metrics dictionary.
    """
    # Import wandb only if needed
    if use_wandb:
        try:
            import wandb
        except ImportError:
            print("Warning: wandb not installed. Disabling wandb logging.")
            use_wandb = False
    driver, env = setup_browser_and_game(
        record_video=record_video, action_delay=action_delay
    )

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    best_score = 0
    best_max_tile = 0
    episode_scores = []
    episode_max_tiles = []
    episode_steps = []
    episode_losses = []

    print(f"Starting training for {num_episodes} episodes...")
    print(
        f"State dim: {env.observation_space.shape[0]}, Action dim: {env.action_space.n}"
    )

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_step = 0
        max_tile = 0
        recent_scores = []
        done = False

        while not done:
            action, probs = agent.select_action(state, return_probs=True)
            next_state, reward, done, info = env.step(
                action, observation=state, probabilities=probs
            )
            agent.store_reward(reward)

            episode_reward += reward
            episode_step += 1
            max_tile = max(max_tile, info["max_tile"])

            # Stagnation detection
            recent_scores.append(info["score"])
            if len(recent_scores) > STAGNATION_WINDOW_SIZE:
                recent_scores.pop(0)

            state = next_state

            # Check for stagnation
            if len(recent_scores) >= STAGNATION_WINDOW_SIZE:
                if recent_scores[-1] - recent_scores[0] <= STAGNATION_THRESHOLD:
                    done = True

            if episode_step > max_steps_per_episode:
                done = True

        # Create video if recording
        if env.controller.record_video and env.controller.video_frames:
            video_path = env.controller._create_video()
            if video_path:
                print(f"Episode {episode + 1} video saved to: {video_path}")

        # Update policy
        loss = agent.update_policy()

        # Record metrics
        final_score = info["score"]
        episode_scores.append(final_score)
        episode_max_tiles.append(max_tile)
        episode_steps.append(episode_step)
        episode_losses.append(loss if loss else 0.0)

        print(
            f"Episode {episode + 1}/{num_episodes} | "
            f"Steps: {episode_step} | "
            f"Score: {final_score} | "
            f"Max Tile: {max_tile} | "
            f"Invalid Moves: {info['invalid_moves']} | "
            f"Loss: {(loss if loss else 0):.4f}"
        )

        # Log to wandb
        if use_wandb:
            wandb.log({
                "episode": episode + 1,
                "score": final_score,
                "max_tile": max_tile,
                "steps": episode_step,
                "invalid_moves": info["invalid_moves"],
                "loss": loss if loss else 0.0,
                "episode_reward": episode_reward,
                "best_score": best_score,
                "best_max_tile": best_max_tile,
                "avg_score_last_10": sum(episode_scores[-10:]) / min(10, len(episode_scores)),
            })

        # Save best model
        if final_score > best_score:
            best_score = final_score
            agent.save_model(os.path.join(models_dir, "best_model_2048.pt"))
            print(f"New best score! Score: {best_score}")

        if max_tile > best_max_tile:
            best_max_tile = max_tile
            print(f"New best max tile! Tile: {best_max_tile}")

        # Restart browser for next episode
        if episode < num_episodes - 1:
            print("Restarting browser for next episode...")
            driver.quit()
            driver, env = setup_browser_and_game(
                record_video=record_video, action_delay=action_delay
            )

    # Save final model and metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(models_dir, f"final_model_2048_{timestamp}.pt")
    agent.save_model(final_model_path)

    metrics = {
        "episode_scores": episode_scores,
        "episode_max_tiles": episode_max_tiles,
        "episode_steps": episode_steps,
        "episode_losses": episode_losses,
        "num_episodes": num_episodes,
        "best_score": best_score,
        "best_max_tile": best_max_tile,
        "timestamp": timestamp,
    }

    metrics_dir = "training_logs"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"training_metrics_2048_{timestamp}.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best Score: {best_score}, Best Max Tile: {best_max_tile}")
    print(f"Metrics saved to: {metrics_path}")

    driver.quit()
    return metrics


def load_and_play(
    agent=None,
    model_path=None,
    num_games=5,
    record_video=False,
    action_delay=0.1,
):
    """Load a trained model and play games with it.

    Args:
        agent: RL agent instance. If None, plays with random actions.
        model_path: Path to saved model weights.
        num_games: Number of games to play.
        record_video: Whether to record gameplay video.
        action_delay: Delay between actions.

    Returns:
        Metrics dictionary with game results.
    """
    driver = setup_browser()
    start_game(driver)

    env = Game2048Env(
        driver,
        save_screenshots=False,
        record_video=record_video,
        action_delay=action_delay,
    )
    print(f"Decision frequency: ~{1/action_delay:.1f} Hz")

    if agent and model_path:
        agent.load_model(model_path)
        print(f"Loaded model from {model_path}")
    elif not agent:
        print("Playing with random actions")

    print(f"\nPlaying {num_games} games...")

    game_scores = []
    game_max_tiles = []
    game_steps = []
    best_score = 0
    best_max_tile = 0

    for game in range(num_games):
        state = env.reset()
        steps = 0
        max_tile = 0
        done = False

        while not done:
            if agent:
                action, probs = agent.select_action(state, return_probs=True)
                next_state, reward, done, info = env.step(
                    action, observation=state, probabilities=probs
                )
            else:
                # Random action
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)

            steps += 1
            max_tile = max(max_tile, info["max_tile"])
            state = next_state

            if steps > 2000:
                done = True

        final_score = info["score"]
        game_scores.append(final_score)
        game_max_tiles.append(max_tile)
        game_steps.append(steps)
        best_score = max(best_score, final_score)
        best_max_tile = max(best_max_tile, max_tile)

        print(
            f"Game {game + 1}/{num_games} | "
            f"Steps: {steps} | "
            f"Score: {final_score} | "
            f"Max Tile: {max_tile}"
        )

        # Save video if recording
        if env.controller.record_video and env.controller.video_frames:
            video_path = env.controller._create_video()
            if video_path:
                print(f"Game {game + 1} video saved to: {video_path}")

        # Clear agent buffer if needed
        if agent and hasattr(agent, "clear_buffer"):
            agent.clear_buffer()

    # Save metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = {
        "model_path": model_path,
        "num_games": num_games,
        "game_scores": game_scores,
        "game_max_tiles": game_max_tiles,
        "game_steps": game_steps,
        "best_score": best_score,
        "best_max_tile": best_max_tile,
        "avg_score": np.mean(game_scores),
        "avg_max_tile": np.mean(game_max_tiles),
        "timestamp": timestamp,
    }

    metrics_dir = "inference_logs"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"inference_metrics_2048_{timestamp}.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nInference metrics saved to: {metrics_path}")
    print(f"Best Score: {best_score} | Best Max Tile: {best_max_tile}")
    print(
        f"Avg Score: {np.mean(game_scores):.1f} | Avg Max Tile: {np.mean(game_max_tiles):.1f}"
    )

    driver.quit()
    return metrics
