"""RL agents for 2048 game.

Provides agent implementations including:
- RandomAgent: Baseline random action selection
- REINFORCEAgent: Policy gradient agent
- A2CAgent: Actor-Critic agent with N-step updates
"""

import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from .utils import NUM_ACTIONS, OBSERVATION_DIM


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    @abstractmethod
    def select_action(self, state, return_probs=False):
        """Select an action given the current state.

        Args:
            state: Current observation.
            return_probs: Whether to return action probabilities.

        Returns:
            action: Selected action index.
            probs: Action probabilities (if return_probs=True).
        """
        pass

    @abstractmethod
    def store_reward(self, reward):
        """Store reward for the last action.

        Args:
            reward: Reward received.
        """
        pass

    @abstractmethod
    def update_policy(self):
        """Update the policy based on collected experience.

        Returns:
            Loss value (or None if no update performed).
        """
        pass

    @abstractmethod
    def save_model(self, path):
        """Save model to disk.

        Args:
            path: File path to save model.
        """
        pass

    @abstractmethod
    def load_model(self, path):
        """Load model from disk.

        Args:
            path: File path to load model from.
        """
        pass

    def clear_buffer(self):
        """Clear any stored experience. Override if needed."""
        pass


class RandomAgent(BaseAgent):
    """Agent that selects random actions. Useful as a baseline."""

    def __init__(self, num_actions=NUM_ACTIONS):
        """Initialize random agent.

        Args:
            num_actions: Number of possible actions.
        """
        self.num_actions = num_actions
        self.rewards = []

    def select_action(self, state, return_probs=False):
        """Select a random action.

        Args:
            state: Current observation (ignored).
            return_probs: Whether to return action probabilities.

        Returns:
            action: Random action index.
            probs: Uniform probabilities (if return_probs=True).
        """
        action = np.random.randint(0, self.num_actions)
        probs = np.ones(self.num_actions) / self.num_actions

        if return_probs:
            return action, probs
        return action

    def store_reward(self, reward):
        """Store reward (does nothing for random agent)."""
        self.rewards.append(reward)

    def update_policy(self):
        """No policy to update for random agent."""
        self.rewards = []
        return None

    def save_model(self, path):
        """Nothing to save for random agent."""
        pass

    def load_model(self, path):
        """Nothing to load for random agent."""
        pass

    def clear_buffer(self):
        """Clear stored rewards."""
        self.rewards = []


class PolicyNetwork(nn.Module):
    """Neural network for policy (actor) in RL agents."""

    def __init__(
        self, state_dim=OBSERVATION_DIM, action_dim=NUM_ACTIONS, hidden_dim=128
    ):
        """Initialize policy network.

        Args:
            state_dim: Dimension of state/observation space.
            action_dim: Number of actions.
            hidden_dim: Size of hidden layers.
        """
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input state tensor.

        Returns:
            Action logits.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ValueNetwork(nn.Module):
    """Neural network for value function (critic) in A2C."""

    def __init__(self, state_dim=OBSERVATION_DIM, hidden_dim=128):
        """Initialize value network.

        Args:
            state_dim: Dimension of state/observation space.
            hidden_dim: Size of hidden layers.
        """
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input state tensor.

        Returns:
            State value estimate.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class REINFORCEAgent(BaseAgent):
    """REINFORCE policy gradient agent.

    Updates policy at the end of each episode using collected rewards.
    """

    def __init__(
        self,
        state_dim=OBSERVATION_DIM,
        action_dim=NUM_ACTIONS,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        entropy_coef=0.01,
    ):
        """Initialize REINFORCE agent.

        Args:
            state_dim: Dimension of state/observation space.
            action_dim: Number of actions.
            hidden_dim: Size of hidden layers.
            lr: Learning rate.
            gamma: Discount factor.
            entropy_coef: Entropy bonus coefficient for exploration.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Episode buffers
        self.log_probs = []
        self.rewards = []
        self.entropies = []

    def select_action(self, state, return_probs=False):
        """Select action using the policy network.

        Args:
            state: Current observation.
            return_probs: Whether to return action probabilities.

        Returns:
            action: Selected action index.
            probs: Action probabilities (if return_probs=True).
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.policy(state_tensor)
            probs = F.softmax(logits, dim=-1)

        dist = Categorical(probs)
        action = dist.sample()

        # Store log probability and entropy for training
        self.log_probs.append(dist.log_prob(action))
        self.entropies.append(dist.entropy())

        action_idx = action.item()
        probs_np = probs.cpu().numpy().flatten()

        if return_probs:
            return action_idx, probs_np
        return action_idx

    def store_reward(self, reward):
        """Store reward for the current step.

        Args:
            reward: Reward received.
        """
        self.rewards.append(reward)

    def update_policy(self):
        """Update policy using REINFORCE algorithm.

        Returns:
            Loss value.
        """
        if len(self.rewards) == 0:
            return None

        # Calculate discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculate policy loss
        policy_loss = 0
        entropy_loss = 0
        for log_prob, entropy, G in zip(self.log_probs, self.entropies, returns):
            policy_loss -= log_prob * G
            entropy_loss -= entropy

        loss = policy_loss + self.entropy_coef * entropy_loss

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        # Clear buffers
        self.clear_buffer()

        return loss.item()

    def clear_buffer(self):
        """Clear episode buffers."""
        self.log_probs = []
        self.rewards = []
        self.entropies = []

    def save_model(self, path):
        """Save model weights.

        Args:
            path: File path to save model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_model(self, path):
        """Load model weights.

        Args:
            path: File path to load model from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class A2CAgent(BaseAgent):
    """Advantage Actor-Critic (A2C) agent.

    Uses separate actor (policy) and critic (value) networks with
    N-step returns for more stable training on long episodes.
    """

    def __init__(
        self,
        state_dim=OBSERVATION_DIM,
        action_dim=NUM_ACTIONS,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        entropy_coef=0.01,
        value_coef=0.5,
        n_steps=32,
    ):
        """Initialize A2C agent.

        Args:
            state_dim: Dimension of state/observation space.
            action_dim: Number of actions.
            hidden_dim: Size of hidden layers.
            lr: Learning rate.
            gamma: Discount factor.
            entropy_coef: Entropy bonus coefficient.
            value_coef: Value loss coefficient.
            n_steps: Number of steps for N-step returns.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_steps = n_steps

        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = ValueNetwork(state_dim, hidden_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Step buffers - store raw data, recompute gradients during update
        self.states = []
        self.actions = []
        self.rewards = []

        self.accumulated_loss = 0.0
        self.update_count = 0

    def select_action(self, state, return_probs=False):
        """Select action using the actor network.

        Args:
            state: Current observation.
            return_probs: Whether to return action probabilities.

        Returns:
            action: Selected action index.
            probs: Action probabilities (if return_probs=True).
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.actor(state_tensor)
            probs = F.softmax(logits, dim=-1)

        dist = Categorical(probs)
        action = dist.sample()
        action_idx = action.item()

        # Store state and action for training (recompute gradients later)
        self.states.append(state)
        self.actions.append(action_idx)

        probs_np = probs.cpu().numpy().flatten()

        if return_probs:
            return action_idx, probs_np
        return action_idx

    def store_reward(self, reward):
        """Store reward and trigger N-step update if needed.

        Args:
            reward: Reward received.
        """
        self.rewards.append(reward)

        # Perform N-step update
        if len(self.rewards) >= self.n_steps:
            loss = self._update_n_step()
            if loss is not None:
                self.accumulated_loss += loss
                self.update_count += 1

    def _update_n_step(self):
        """Perform N-step A2C update.

        Returns:
            Loss value.
        """
        if len(self.rewards) == 0:
            return None

        # Get data for this update
        n = len(self.rewards)
        states = self.states[:n]
        actions = self.actions[:n]
        rewards = self.rewards

        # Bootstrap value for last state (if available)
        if len(self.states) > n:
            next_state = torch.FloatTensor(self.states[n]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_value = self.critic(next_state).item()
        else:
            next_value = 0

        # Compute returns
        returns = []
        G = next_value
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Recompute log_probs, values, entropies WITH gradients
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)

        # Forward pass with gradients
        logits = self.actor(states_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        log_probs_tensor = dist.log_prob(actions_tensor)
        entropies_tensor = dist.entropy()
        values_tensor = self.critic(states_tensor).squeeze()

        # Advantage = Returns - Values (detach values for advantage computation)
        advantages = returns - values_tensor.detach()

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor loss (policy gradient with advantage)
        actor_loss = -(log_probs_tensor * advantages).mean()
        entropy_loss = -entropies_tensor.mean()

        # Critic loss (MSE between values and returns)
        critic_loss = F.mse_loss(values_tensor, returns)

        # Total loss
        total_loss = (
            actor_loss
            + self.value_coef * critic_loss
            + self.entropy_coef * entropy_loss
        )

        # Update networks
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Clear processed steps
        self.states = self.states[n:]
        self.actions = self.actions[n:]
        self.rewards = []

        return total_loss.item()

    def update_policy(self):
        """Final update at end of episode for remaining steps.

        Returns:
            Average loss over all updates this episode.
        """
        # Update any remaining steps
        if len(self.rewards) > 0:
            loss = self._update_n_step()
            if loss is not None:
                self.accumulated_loss += loss
                self.update_count += 1

        # Return average loss
        avg_loss = (
            self.accumulated_loss / self.update_count if self.update_count > 0 else 0.0
        )

        # Reset for next episode
        self.clear_buffer()
        self.accumulated_loss = 0.0
        self.update_count = 0

        return avg_loss

    def clear_buffer(self):
        """Clear all buffers."""
        self.states = []
        self.actions = []
        self.rewards = []

    def save_model(self, path):
        """Save model weights.

        Args:
            path: File path to save model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load_model(self, path):
        """Load model weights.

        Args:
            path: File path to load model from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        if "actor_optimizer_state_dict" in checkpoint:
            self.actor_optimizer.load_state_dict(
                checkpoint["actor_optimizer_state_dict"]
            )
        if "critic_optimizer_state_dict" in checkpoint:
            self.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer_state_dict"]
            )
