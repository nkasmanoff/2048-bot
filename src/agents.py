"""RL agents for 2048 game.

Provides agent implementations including:
- RandomAgent: Baseline random action selection
- REINFORCEAgent: Policy gradient agent
- A2CAgent: Actor-Critic agent with N-step updates
- PPOAgent: Proximal Policy Optimization agent (recommended)
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
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input state tensor.

        Returns:
            Action logits.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
                
        return self.fc5(x)


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
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input state tensor.

        Returns:
            State value estimate.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
                
        return self.fc5(x)


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


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization (PPO) agent.

    Uses clipped surrogate objective for stable policy updates with
    multiple epochs of minibatch updates on collected experience.
    """

    def __init__(
        self,
        state_dim=OBSERVATION_DIM,
        action_dim=NUM_ACTIONS,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        n_steps=128,
        n_epochs=4,
        batch_size=32,
    ):
        """Initialize PPO agent.

        Args:
            state_dim: Dimension of state/observation space.
            action_dim: Number of actions.
            hidden_dim: Size of hidden layers.
            lr: Learning rate.
            gamma: Discount factor.
            gae_lambda: Lambda for GAE (Generalized Advantage Estimation).
            clip_epsilon: Clipping parameter for PPO objective.
            entropy_coef: Entropy bonus coefficient.
            value_coef: Value loss coefficient.
            max_grad_norm: Maximum gradient norm for clipping.
            n_steps: Number of steps to collect before update.
            n_epochs: Number of epochs for each update.
            batch_size: Minibatch size for updates.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = ValueNetwork(state_dim, hidden_dim).to(self.device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        # Rollout buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

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
            value = self.critic(state_tensor)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action_idx = action.item()

        # Store experience
        self.states.append(state)
        self.actions.append(action_idx)
        self.values.append(value.item())
        self.log_probs.append(log_prob.item())

        probs_np = probs.cpu().numpy().flatten()

        if return_probs:
            return action_idx, probs_np
        return action_idx

    def store_reward(self, reward, done=False):
        """Store reward and done flag, trigger update if buffer full.

        Args:
            reward: Reward received.
            done: Whether episode ended.
        """
        self.rewards.append(reward)
        self.dones.append(done)

        # Perform PPO update when buffer is full
        if len(self.rewards) >= self.n_steps:
            loss = self._update_ppo()
            if loss is not None:
                self.accumulated_loss += loss
                self.update_count += 1

    def _compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: List of rewards.
            values: List of value estimates.
            dones: List of done flags.
            next_value: Bootstrap value for last state.

        Returns:
            advantages: Tensor of advantages.
            returns: Tensor of returns.
        """
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            # If done, no bootstrapping
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float32).to(self.device)

        return advantages, returns

    def _update_ppo(self):
        """Perform PPO update with multiple epochs.

        Returns:
            Average loss over all updates.
        """
        if len(self.rewards) == 0:
            return None

        # Get next value for GAE bootstrap
        if len(self.states) > len(self.rewards):
            next_state = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_value = self.critic(next_state).item()
        else:
            next_value = 0

        # Use only the data we have rewards for
        n = len(self.rewards)
        states = self.states[:n]
        actions = self.actions[:n]
        old_log_probs = self.log_probs[:n]
        values = self.values[:n]
        rewards = self.rewards
        dones = self.dones

        # Compute GAE
        advantages, returns = self._compute_gae(rewards, values, dones, next_value)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)

        # PPO update with multiple epochs
        total_loss = 0.0
        num_updates = 0
        n_samples = len(states)

        for _ in range(self.n_epochs):
            # Generate random permutation for minibatches
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_indices = indices[start:end]

                # Get batch data
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Forward pass
                logits = self.actor(batch_states)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)

                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                new_values = self.critic(batch_states).squeeze()

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped or simple MSE)
                value_loss = F.mse_loss(new_values, batch_returns)

                # Total loss
                loss = (
                    actor_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_loss += loss.item()
                num_updates += 1

        # Clear processed experience
        self.states = self.states[n:]
        self.actions = self.actions[n:]
        self.values = self.values[n:]
        self.log_probs = self.log_probs[n:]
        self.rewards = []
        self.dones = []

        return total_loss / num_updates if num_updates > 0 else 0.0

    def update_policy(self):
        """Final update at end of episode for remaining steps.

        Returns:
            Average loss over all updates this episode.
        """
        # Update any remaining steps
        if len(self.rewards) > 0:
            loss = self._update_ppo()
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
        """Clear all rollout buffers."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

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
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
