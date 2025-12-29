"""2048 game controller using Selenium WebDriver.

Provides the Game2048Controller class for interacting with the game,
including movement, state extraction, and video recording.
"""

import io
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from .utils import ACTION_KEYS, ACTION_NAMES


class Game2048Controller:
    """Controller for interacting with the 2048 game via Selenium."""

    def __init__(self, driver, save_screenshots=False, record_video=False):
        """Initialize the 2048 controller.

        Args:
            driver: Selenium WebDriver instance.
            save_screenshots: Whether to save screenshots to disk.
            record_video: Whether to record gameplay video.
        """
        self.driver = driver
        self.save_screenshots = save_screenshots
        self.record_video = record_video
        self.frame_count = 0
        self.screenshot_folder = None
        self.images_folder = None
        self.game_start_time = datetime.now()
        self.video_frames = [] if record_video else None
        self.video_annotations = [] if record_video else None
        self.video_fps = 10
        self.game_log = {"game_start": self.game_start_time.isoformat(), "frames": []}
        self.previous_score = 0

        if self.save_screenshots or self.record_video:
            self._setup_screenshot_folder()

    def _setup_screenshot_folder(self):
        """Create folders for screenshots and video output."""
        games_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "games"
        )
        os.makedirs(games_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.screenshot_folder = os.path.join(games_folder, timestamp)
        os.makedirs(self.screenshot_folder, exist_ok=True)

        if self.save_screenshots:
            self.images_folder = os.path.join(self.screenshot_folder, "images")
            os.makedirs(self.images_folder, exist_ok=True)
            print(f"Screenshots will be saved to: {self.images_folder}")
        else:
            self.images_folder = None

        if self.record_video:
            print(f"Video will be saved to: {self.screenshot_folder}")

    def capture_screenshot(self, action_type="move", action=None):
        """Capture a screenshot with metadata and log game state."""
        if (
            not (self.save_screenshots or self.record_video)
            or not self.screenshot_folder
        ):
            return

        action_name = ACTION_NAMES[action] if action is not None else "none"
        filename = f"frame_{self.frame_count:06d}_action_{action_name}.png"
        filepath = (
            os.path.join(self.images_folder, filename) if self.images_folder else None
        )

        game_state = self.get_game_state()
        screenshot_png = self.driver.get_screenshot_as_png()

        if self.save_screenshots and filepath:
            with open(filepath, "wb") as f:
                f.write(screenshot_png)

        if self.record_video:
            img = Image.open(io.BytesIO(screenshot_png))
            frame = np.array(img)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_frames.append(frame_bgr)
            self.video_annotations.append(None)

        current_time = datetime.now()
        elapsed_time = (current_time - self.game_start_time).total_seconds()

        frame_state = {
            "frame_id": self.frame_count,
            "timestamp": current_time.isoformat(),
            "elapsed_seconds": round(elapsed_time, 3),
            "image_file": f"images/{filename}" if self.save_screenshots else None,
            "state": {
                "board": game_state.get("board") if game_state else None,
                "score": game_state.get("score", 0) if game_state else 0,
                "is_game_over": (
                    game_state.get("is_game_over", False) if game_state else False
                ),
            },
            "action": {
                "type": action_type,
                "direction": action_name,
            },
        }

        if self.save_screenshots:
            self.game_log["frames"].append(frame_state)

        self.frame_count += 1

        if self.save_screenshots and self.frame_count % 10 == 0:
            self.save_game_log()

    def save_game_log(self):
        """Save the game log to a JSON file."""
        if not self.screenshot_folder:
            return

        game_state = self.get_game_state()
        self.game_log["game_end"] = datetime.now().isoformat()
        self.game_log["total_frames"] = self.frame_count
        self.game_log["duration_seconds"] = (
            datetime.now() - self.game_start_time
        ).total_seconds()
        self.game_log["final_score"] = game_state.get("score", 0) if game_state else 0
        self.game_log["max_tile"] = self._get_max_tile(game_state)

        json_path = None
        if self.save_screenshots:
            json_path = os.path.join(self.screenshot_folder, "game_log.json")
            with open(json_path, "w") as f:
                json.dump(self.game_log, f, indent=2)

        if self.record_video and self.video_frames:
            video_path = self._create_video()
            if video_path:
                print(f"Video saved to: {video_path}")

        return json_path

    def _get_max_tile(self, game_state):
        """Get the maximum tile value from the board."""
        if not game_state or not game_state.get("board"):
            return 0
        max_tile = 0
        for row in game_state["board"]:
            for tile in row:
                max_tile = max(max_tile, tile)
        return max_tile

    def _create_video(self):
        """Create a video from collected frames with annotations."""
        if not self.video_frames or not self.screenshot_folder:
            return None

        try:
            height, width = self.video_frames[0].shape[:2]
            print(
                f"Creating video with {len(self.video_frames)} frames at {self.video_fps} FPS"
            )

            video_filename = os.path.join(self.screenshot_folder, "gameplay.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                video_filename, fourcc, self.video_fps, (width, height)
            )

            if not out.isOpened():
                print(f"Error: Failed to open video writer for {video_filename}")
                return None

            for i, frame in enumerate(self.video_frames):
                annotated_frame = self._annotate_frame(frame, i)
                out.write(annotated_frame)
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(self.video_frames)} frames...")

            out.release()

            if os.path.exists(video_filename):
                file_size = os.path.getsize(video_filename)
                print(
                    f"Video created: {video_filename} ({file_size / 1024 / 1024:.2f} MB)"
                )
            else:
                return None

            self.video_frames = []
            self.video_annotations = []
            return video_filename

        except Exception as e:
            print(f"Error creating video: {e}")
            return None

    def _annotate_frame(self, frame, frame_idx):
        """Add annotations to a video frame."""
        if not self.video_annotations or frame_idx >= len(self.video_annotations):
            return frame

        annotation = self.video_annotations[frame_idx]
        if not annotation:
            return frame

        annotated_frame = frame.copy()
        # Add annotation overlay if needed
        return annotated_frame

    def set_frame_annotation(self, observation=None, probabilities=None, action=None):
        """Set annotation for the most recently captured frame."""
        if (
            self.record_video
            and self.video_annotations
            and len(self.video_annotations) > 0
        ):
            self.video_annotations[-1] = {
                "observation": observation,
                "probabilities": probabilities,
                "action": action,
            }

    def get_game_state(self):
        """Get the current game state including board, score, and game over status."""
        try:
            result = self.driver.execute_script(
                """
                var result = {
                    board: null,
                    score: 0,
                    is_game_over: false,
                    won: false
                };
                
                // Try to get the board state from tile containers
                var tileContainer = document.querySelector('.tile-container');
                if (tileContainer) {
                    // Initialize empty 4x4 board
                    result.board = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]];
                    
                    var tiles = tileContainer.querySelectorAll('.tile');
                    for (var i = 0; i < tiles.length; i++) {
                        var tile = tiles[i];
                        var classes = tile.className;
                        
                        // Extract value from class like "tile-2", "tile-4", etc.
                        var valueMatch = classes.match(/tile-(\\d+)/);
                        var value = valueMatch ? parseInt(valueMatch[1]) : 0;
                        
                        // Extract position from class like "tile-position-1-2"
                        var posMatch = classes.match(/tile-position-(\\d+)-(\\d+)/);
                        if (posMatch && value > 0) {
                            var col = parseInt(posMatch[1]) - 1;  // 1-indexed to 0-indexed
                            var row = parseInt(posMatch[2]) - 1;
                            if (row >= 0 && row < 4 && col >= 0 && col < 4) {
                                // Keep highest value if multiple tiles at same position (during merge animation)
                                if (value > result.board[row][col]) {
                                    result.board[row][col] = value;
                                }
                            }
                        }
                    }
                }
                
                // Get score
                var scoreContainer = document.querySelector('.score-container');
                if (scoreContainer) {
                    var scoreText = scoreContainer.textContent;
                    var scoreMatch = scoreText.match(/^(\\d+)/);
                    if (scoreMatch) {
                        result.score = parseInt(scoreMatch[1]);
                    }
                }
                
                // Check for game over
                var gameMessage = document.querySelector('.game-message');
                if (gameMessage) {
                    if (gameMessage.classList.contains('game-over')) {
                        result.is_game_over = true;
                    }
                    if (gameMessage.classList.contains('game-won')) {
                        result.won = true;
                    }
                }
                
                return result;
            """
            )
            return result
        except Exception as e:
            print(f"Error getting game state: {e}")
            return None

    def get_score(self):
        """Get the current score."""
        state = self.get_game_state()
        return state.get("score", 0) if state else 0

    def get_board(self):
        """Get the current board state as a 4x4 list."""
        state = self.get_game_state()
        return state.get("board") if state else None

    def is_game_over(self):
        """Check if the game is over."""
        state = self.get_game_state()
        return state.get("is_game_over", False) if state else False

    def has_won(self):
        """Check if the player has won (reached 2048)."""
        state = self.get_game_state()
        return state.get("won", False) if state else False

    def move(self, action):
        """Execute a move in the specified direction.

        Args:
            action: Action index (0=Up, 1=Right, 2=Down, 3=Left).

        Returns:
            dict with move result info.
        """
        # Store state before move
        state_before = self.get_game_state()
        score_before = state_before.get("score", 0) if state_before else 0
        board_before = state_before.get("board") if state_before else None

        # Send key press
        key = ACTION_KEYS[action]
        body = self.driver.find_element(By.TAG_NAME, "body")

        if key == "ArrowUp":
            body.send_keys(Keys.ARROW_UP)
        elif key == "ArrowRight":
            body.send_keys(Keys.ARROW_RIGHT)
        elif key == "ArrowDown":
            body.send_keys(Keys.ARROW_DOWN)
        elif key == "ArrowLeft":
            body.send_keys(Keys.ARROW_LEFT)

        # Brief wait for animation
        time.sleep(0.15)

        # Get state after move
        state_after = self.get_game_state()
        score_after = state_after.get("score", 0) if state_after else 0
        board_after = state_after.get("board") if state_after else None

        # Check if move was valid (board changed)
        move_valid = (
            board_before != board_after if board_before and board_after else True
        )
        score_gained = score_after - score_before

        self.capture_screenshot(action_type="move", action=action)

        return {
            "action": action,
            "action_name": ACTION_NAMES[action],
            "score_before": score_before,
            "score_after": score_after,
            "score_gained": score_gained,
            "move_valid": move_valid,
            "is_game_over": (
                state_after.get("is_game_over", False) if state_after else False
            ),
        }

    def reset_game(self):
        """Reset the game by clicking the New Game button."""
        try:
            # Try clicking the restart button
            restart_buttons = self.driver.find_elements(By.CLASS_NAME, "restart-button")
            for button in restart_buttons:
                if button.is_displayed():
                    button.click()
                    time.sleep(0.5)
                    self.previous_score = 0
                    self.game_start_time = datetime.now()
                    return True

            # Try "Try again" button (shown on game over)
            try_again = self.driver.find_elements(By.CLASS_NAME, "retry-button")
            for button in try_again:
                if button.is_displayed():
                    button.click()
                    time.sleep(0.5)
                    self.previous_score = 0
                    self.game_start_time = datetime.now()
                    return True

        except Exception as e:
            print(f"Error resetting game: {e}")
            return False

        return False

    def capture_frame_only(self):
        """Capture a frame for video without logging."""
        if not self.record_video or not self.screenshot_folder:
            return

        try:
            screenshot_png = self.driver.get_screenshot_as_png()
            img = Image.open(io.BytesIO(screenshot_png))
            frame = np.array(img)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_frames.append(frame_bgr)
            self.video_annotations.append(None)
            self.frame_count += 1
        except Exception as e:
            print(f"Error capturing frame: {e}")
