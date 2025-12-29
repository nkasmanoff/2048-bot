"""LLM-based agent for 2048 game using OpenAI API format.

Provides LLMAgent that uses language models to play 2048 by:
- Receiving game state as text
- Reasoning about moves in <thinking> tags
- Outputting actions in <action> tags
- Logging thoughts and actions during gameplay
"""

import os
import re
from datetime import datetime

import numpy as np
from openai import OpenAI

from .agents import BaseAgent
from .utils import NUM_ACTIONS, ACTION_NAMES


class LLMAgent(BaseAgent):
    """Agent that uses an LLM to play 2048.

    The agent sends the current game state to an LLM and parses its
    response to extract the chosen action. Supports any OpenAI-compatible API.
    """

    def __init__(
        self,
        model_name=None,
        api_key=None,
        temperature=0.7,
        log_dir="llm_logs",
        verbose=True,
        base_url=None,
    ):
        """Initialize LLM agent.

        Args:
            model_name: Name of the model to use (or None to use LLM_MODEL env var).
            api_key: API key (or None to use OPENROUTER_API_KEY/OPENAI_API_KEY env var).
            base_url: Custom base URL for OpenAI-compatible APIs (or None to use LLM_BASE_URL env var).
            temperature: Sampling temperature (0.0 to 1.0).
            log_dir: Directory to save interaction logs.
            verbose: Whether to print thoughts and actions to console.
        """
        # Get model name from env if not provided
        if model_name is None:
            model_name = os.getenv("LLM_MODEL")
            if not model_name:
                model_name = "gpt-4o-mini"  # Fallback default
                if verbose:
                    print(f"⚠️  LLM_MODEL not set, using default: {model_name}")

        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose

        # Get API key from env if not provided
        if api_key is None:
            # Try OpenRouter first, then OpenAI
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "API key must be provided via api_key parameter or "
                    "OPENROUTER_API_KEY/OPENAI_API_KEY environment variable"
                )

        # Get base URL from env if not provided
        if base_url is None:
            base_url = os.getenv("LLM_BASE_URL", "")

        # Initialize OpenAI client
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

        # Setup logging
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            log_dir, f"llm_game_{model_name.replace('/', '_')}_{timestamp}.log"
        )

        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        self.rewards = []

        # System prompt that explains the game and output format
        self.system_prompt = """You are playing the game 2048. Your goal is to merge tiles to create higher values, ultimately reaching 2048 or beyond.

Game Rules:
- The game is played on a 4×4 grid
- You can move tiles in four directions: Up, Right, Down, Left
- When tiles with the same number touch, they merge into their sum
- After each move, a new tile (2 or 4) appears in an empty spot
- The game ends when no valid moves remain
- You can merge tiles when they are the same number and are adjacent to each other

Strategy Tips:
- Keep high-value tiles in one corner (usually a corner)
- Try to maintain tiles in descending order
- Avoid moves that break your tile organization
- Consider both immediate gains and future board state

Your Response Format:
You must respond with your reasoning wrapped in <thinking> tags, followed by your chosen action in <action> tags.

Example:
<thinking>
The board has high tiles in the top-left corner. I should move left to keep them together and create merge opportunities. Moving down would scatter the tiles.
</thinking>
<action>Left</action>

Available actions: Up, Right, Down, Left

Now, analyze the board state and choose your next move."""

        self._log("=== LLM Agent Initialized ===")
        self._log(f"Model: {model_name}")
        self._log(f"Temperature: {temperature}")
        self._log(f"Log file: {self.log_file}")
        self._log("")

    def _log(self, message):
        """Write message to log file and optionally print to console.

        Args:
            message: Message to log.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

        if self.verbose:
            print(log_line)

    def _format_board_state(self, state):
        """Convert game state to human-readable text for LLM.

        Args:
            state: Observation array from environment.

        Returns:
            Formatted string describing the game state.
        """
        # Extract board tiles (first 16 values, denormalized)
        board_values = []
        for i in range(16):
            val = state[i]
            if val == 0:
                tile = 0
            else:
                # Reverse normalization: val = log2(tile) / 15.0
                log_val = val * 15.0
                tile = int(2**log_val)
            board_values.append(tile)

        # Format as 4x4 grid
        board_str = "Current Board:\n"
        board_str += "┌" + "─" * 25 + "┐\n"
        for row in range(4):
            row_vals = board_values[row * 4 : (row + 1) * 4]
            row_str = "│ " + " ".join(f"{v:4d}" for v in row_vals) + " │\n"
            board_str += row_str
        board_str += "└" + "─" * 25 + "┘\n"

        # Extract metadata
        can_move_up = bool(state[16])
        can_move_right = bool(state[17])
        can_move_down = bool(state[18])
        can_move_left = bool(state[19])
        empty_cells = int(state[20] * 16)
        max_tile = int(2 ** (state[21] * 15.0)) if state[21] > 0 else 0
        score_norm = state[22]
        estimated_score = int(10 ** (score_norm * 6.0)) - 1

        # Add metadata
        metadata = "\nGame Info:\n"
        metadata += f"- Max tile: {max_tile}\n"
        metadata += f"- Empty cells: {empty_cells}\n"
        metadata += f"- Estimated score: ~{estimated_score}\n"
        metadata += f"- Step: {self.step_count}\n"
        metadata += "\nAvailable moves:\n"
        metadata += f"- Up: {'✓' if can_move_up else '✗'}\n"
        metadata += f"- Right: {'✓' if can_move_right else '✗'}\n"
        metadata += f"- Down: {'✓' if can_move_down else '✗'}\n"
        metadata += f"- Left: {'✓' if can_move_left else '✗'}\n"

        return board_str + metadata

    def _query_llm(self, state):
        """Query the LLM with the current game state.

        Args:
            state: Observation array from environment.

        Returns:
            Tuple of (thinking, action_name) extracted from LLM response.
        """
        # Format the board state
        board_text = self._format_board_state(state)

        # Create the prompt
        user_message = f"{board_text}\n\nWhat is your next move?"

        # Call the LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=500,
            )

            response_text = response.choices[0].message.content

            # Extract thinking and action
            thinking = self._extract_tag_content(response_text, "thinking")
            action_name = self._extract_tag_content(response_text, "action")

            if not action_name:
                # Try to find action name without tags
                for name in ACTION_NAMES:
                    if name.lower() in response_text.lower():
                        action_name = name
                        break

            if not thinking:
                thinking = "(No thinking provided)"

            if not action_name:
                self._log(
                    f"⚠️ Could not parse action from response: {response_text[:200]}"
                )
                action_name = "Up"  # Default fallback

            return thinking, action_name

        except Exception as e:
            self._log(f"❌ Error querying LLM: {str(e)}")
            return f"Error: {str(e)}", "Up"

    def _extract_tag_content(self, text, tag_name):
        """Extract content from XML-style tags.

        Args:
            text: Full text containing tags.
            tag_name: Name of the tag to extract (without < >).

        Returns:
            Content inside the tags, or None if not found.
        """
        pattern = f"<{tag_name}>(.*?)</{tag_name}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def select_action(self, state, return_probs=False):
        """Select action using LLM reasoning.

        Args:
            state: Current observation array.
            return_probs: Whether to return action probabilities.

        Returns:
            action: Selected action index.
            probs: Uniform probabilities (if return_probs=True).
        """
        self.step_count += 1

        self._log(f"\n{'='*50}")
        self._log(f"Step {self.step_count}")
        self._log(f"{'='*50}")

        # Query LLM
        thinking, action_name = self._query_llm(state)

        # Log thinking
        self._log("\n💭 LLM Thinking:")
        for line in thinking.split("\n"):
            self._log(f"   {line}")

        # Convert action name to index
        action_name_upper = action_name.strip().capitalize()
        if action_name_upper in ACTION_NAMES:
            action = ACTION_NAMES.index(action_name_upper)
        else:
            self._log(f"⚠️ Invalid action '{action_name}', defaulting to Up")
            action = 0

        # Log action
        self._log(f"\n🎮 Action Selected: {ACTION_NAMES[action]} (index: {action})")

        # Return uniform probabilities (LLM doesn't provide true probabilities)
        probs = np.ones(NUM_ACTIONS) / NUM_ACTIONS

        if return_probs:
            return action, probs
        return action

    def store_reward(self, reward):
        """Store reward for the last action.

        Args:
            reward: Reward received.
        """
        self.rewards.append(reward)
        self._log(f"💰 Reward: {reward:.2f}")

    def update_policy(self):
        """No policy update for LLM agent.

        Returns:
            None (LLM doesn't learn during gameplay).
        """
        self.episode_count += 1
        total_reward = sum(self.rewards)

        self._log(f"\n{'='*50}")
        self._log(f"Episode {self.episode_count} Complete")
        self._log(f"{'='*50}")
        self._log(f"Total steps: {self.step_count}")
        self._log(f"Total reward: {total_reward:.2f}")
        self._log(f"Average reward: {total_reward / max(self.step_count, 1):.2f}")
        self._log("")

        # Reset for next episode
        self.rewards = []
        self.step_count = 0

        return None

    def clear_buffer(self):
        """Clear episode buffers."""
        self.rewards = []

    def save_model(self, path):
        """LLM agent has no model to save.

        Args:
            path: File path (ignored).
        """
        self._log("Note: LLM agent has no model to save")

    def load_model(self, path):
        """LLM agent has no model to load.

        Args:
            path: File path (ignored).
        """
        self._log("Note: LLM agent has no model to load")


def create_llm_agent(verbose=True):
    """Create an LLM agent using environment variables for configuration.

    Reads from .env file:
        - LLM_MODEL: Model name (e.g., 'gpt-4o-mini', 'allenai/olmo-3-7b-instruct')
        - OPENROUTER_API_KEY or OPENAI_API_KEY: API key
        - LLM_BASE_URL: Optional base URL for OpenAI-compatible APIs

    Args:
        verbose: Whether to print logs to console.

    Returns:
        Configured LLMAgent instance.
    """
    # Load environment variables
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        if verbose:
            print(
                "⚠️  python-dotenv not installed, using existing environment variables"
            )

    # Get configuration from environment
    model_name = os.getenv("LLM_MODEL")
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "API key not found. Please set OPENROUTER_API_KEY or OPENAI_API_KEY "
            "in your .env file or environment"
        )

    if verbose:
        print(f"\n✓ Using model: {model_name or 'gpt-4o-mini (default)'}")
        if base_url:
            print(f"✓ Using base URL: {base_url}")

    agent = LLMAgent(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        verbose=verbose,
    )

    if verbose:
        print("✓ LLM Agent created successfully")
        print(f"✓ Logs will be saved to: {agent.log_file}\n")

    return agent
