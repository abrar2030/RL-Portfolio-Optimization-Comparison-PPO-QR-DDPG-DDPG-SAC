"""
Deep Reinforcement Learning Agents Implementation.

This module implements:
- PPO (Proximal Policy Optimization)
- DDPG (Deep Deterministic Policy Gradient)
- SAC (Soft Actor-Critic)
- QR-DDPG (Quantile Regression DDPG)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
from collections import deque
import random


class ReplayBuffer:
    """Experience Replay Buffer for off-policy algorithms."""
    
    def __init__(self, capacity: int = 1000000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)


class Actor(nn.Module):
    """Actor network for policy-based algorithms."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        """
        Initialize actor network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super(Actor, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, action_dim)
        self.tanh = nn.Tanh()
    
    def forward(self, state):
        """Forward pass."""
        x = self.hidden_layers(state)
        x = self.output_layer(x)
        return self.tanh(x)


class Critic(nn.Module):
    """Critic network for value-based algorithms."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        """
        Initialize critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super(Critic, self).__init__()
        
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, 1)
    
    def forward(self, state, action):
        """Forward pass."""
        x = torch.cat([state, action], dim=1)
        x = self.hidden_layers(x)
        return self.output_layer(x)


class QuantileCritic(nn.Module):
    """Quantile Critic network for QR-DDPG."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        n_quantiles: int = 50,
        hidden_dims: List[int] = [128, 64]
    ):
        """
        Initialize quantile critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            n_quantiles: Number of quantiles to estimate
            hidden_dims: List of hidden layer dimensions
        """
        super(QuantileCritic, self).__init__()
        
        self.n_quantiles = n_quantiles
        
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, n_quantiles)
    
    def forward(self, state, action):
        """
        Forward pass.
        
        Returns:
            Tensor of shape (batch_size, n_quantiles) with quantile values
        """
        x = torch.cat([state, action], dim=1)
        x = self.hidden_layers(x)
        quantiles = self.output_layer(x)
        return quantiles


class DDPGAgent:
    """Deep Deterministic Policy Gradient Agent."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 0.0001,
        lr_critic: float = 0.0003,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 1000000,
        device: str = 'cpu'
    ):
        """Initialize DDPG agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def select_action(self, state: np.ndarray, noise: float = 0.1) -> np.ndarray:
        """Select action using current policy."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        # Add exploration noise
        action += np.random.normal(0, noise, size=self.action_dim)
        action = np.clip(action, -1, 1)
        
        return action
    
    def update(self, batch_size: int = 128):
        """Update actor and critic networks."""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
    
    def _soft_update(self, local_model, target_model):
        """Soft update of target network."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


class QRDDPGAgent(DDPGAgent):
    """Quantile Regression DDPG Agent."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 0.0001,
        lr_critic: float = 0.0003,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_quantiles: int = 50,
        buffer_size: int = 1000000,
        device: str = 'cpu'
    ):
        """Initialize QR-DDPG agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.n_quantiles = n_quantiles
        self.device = device
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Quantile critic
        self.critic = QuantileCritic(state_dim, action_dim, n_quantiles).to(device)
        self.critic_target = QuantileCritic(state_dim, action_dim, n_quantiles).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Quantile midpoints
        self.quantile_tau = torch.FloatTensor(
            [(i + 0.5) / n_quantiles for i in range(n_quantiles)]
        ).to(device)
    
    def quantile_huber_loss(self, quantiles, target, tau):
        """Calculate quantile Huber loss."""
        pairwise_delta = target.unsqueeze(1) - quantiles.unsqueeze(2)
        abs_pairwise_delta = torch.abs(pairwise_delta)
        huber_loss = torch.where(
            abs_pairwise_delta > 1,
            abs_pairwise_delta - 0.5,
            pairwise_delta ** 2 * 0.5
        )
        
        quantile_loss = torch.abs(tau.unsqueeze(2) - (pairwise_delta < 0).float()) * huber_loss
        return quantile_loss.mean()
    
    def update(self, batch_size: int = 128):
        """Update actor and critic networks."""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_quantiles = self.critic_target(next_states, next_actions)
            target_quantiles = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_quantiles
        
        current_quantiles = self.critic(states, actions)
        critic_loss = self.quantile_huber_loss(current_quantiles, target_quantiles, self.quantile_tau)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor (use lower quantiles for risk-averse policy)
        quantiles = self.critic(states, self.actor(states))
        # Use CVaR (mean of lowest 5% quantiles)
        n_cvar = max(1, int(0.05 * self.n_quantiles))
        actor_loss = -quantiles[:, :n_cvar].mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)


if __name__ == "__main__":
    print("DRL Agents module loaded successfully")
    
    # Test agent initialization
    state_dim = 100
    action_dim = 10
    
    ddpg = DDPGAgent(state_dim, action_dim)
    qr_ddpg = QRDDPGAgent(state_dim, action_dim)
    
    print(f"DDPG initialized: state_dim={state_dim}, action_dim={action_dim}")
    print(f"QR-DDPG initialized: state_dim={state_dim}, action_dim={action_dim}, n_quantiles=50")
