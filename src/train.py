"""
Main training script for RL agents.

This script:
1. Loads and processes data
2. Creates training environment
3. Trains DRL agents (PPO, DDPG, SAC, QR-DDPG)
4. Evaluates agents
5. Saves results
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor
from environment import PortfolioEnv
from agents import QRDDPGAgent


class TrainDRLAgents:
    """Train and evaluate DRL agents."""
    
    def __init__(self, config_path: str = '../config/config.yaml'):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = self.config['output']['results_dir']
        self.models_dir = self.config['output']['models_dir']
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        print("Training configuration loaded")
    
    def prepare_data(self):
        """Load and process data."""
        print("\n" + "="*50)
        print("STEP 1: Data Preparation")
        print("="*50)
        
        processor = DataProcessor(self.config)
        self.train_data, self.test_data = processor.process_all()
        
        print(f"\nTrain data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        print(f"Number of assets: {self.train_data['tic'].nunique()}")
    
    def create_env(self, data: pd.DataFrame, is_training: bool = True):
        """
        Create portfolio environment.
        
        Args:
            data: Processed DataFrame
            is_training: Whether this is for training or testing
        
        Returns:
            Wrapped environment
        """
        env_config = self.config['environment']
        risk_config = self.config['risk']
        
        env = PortfolioEnv(
            df=data,
            initial_amount=env_config['initial_amount'],
            transaction_cost_pct=env_config['transaction_cost_pct'],
            max_drawdown_penalty=risk_config['max_drawdown_penalty'],
            hmax=env_config['hmax'],
            print_verbosity=env_config['print_verbosity']
        )
        
        # Wrap for Stable-Baselines3
        env = DummyVecEnv([lambda: env])
        
        return env
    
    def train_ppo(self, seed: int = 0):
        """
        Train PPO agent.
        
        Args:
            seed: Random seed
        
        Returns:
            Trained model
        """
        print(f"\nTraining PPO (seed={seed})...")
        
        # Create environment
        env = self.create_env(self.train_data)
        
        # Get hyperparameters
        ppo_config = self.config['models']['ppo']
        training_config = self.config['training']
        
        # Create model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=ppo_config['learning_rate'],
            n_steps=ppo_config['n_steps'],
            batch_size=ppo_config['batch_size'],
            n_epochs=ppo_config['n_epochs'],
            gamma=0.99,
            gae_lambda=ppo_config['gae_lambda'],
            clip_range=ppo_config['clip_range'],
            ent_coef=ppo_config['ent_coef'],
            vf_coef=ppo_config['vf_coef'],
            max_grad_norm=ppo_config['max_grad_norm'],
            policy_kwargs=dict(net_arch=[128, 64]),
            verbose=0,
            seed=seed
        )
        
        # Train
        model.learn(
            total_timesteps=training_config['total_timesteps'],
            log_interval=training_config['log_interval']
        )
        
        # Save model
        model_path = os.path.join(self.models_dir, f'ppo_seed_{seed}')
        model.save(model_path)
        
        return model
    
    def train_ddpg(self, seed: int = 0):
        """
        Train DDPG agent.
        
        Args:
            seed: Random seed
        
        Returns:
            Trained model
        """
        print(f"\nTraining DDPG (seed={seed})...")
        
        # Create environment
        env = self.create_env(self.train_data)
        
        # Get hyperparameters
        ddpg_config = self.config['models']['ddpg']
        training_config = self.config['training']
        
        # Create model
        model = DDPG(
            "MlpPolicy",
            env,
            learning_rate=ddpg_config['learning_rate_actor'],
            buffer_size=ddpg_config['buffer_size'],
            batch_size=ddpg_config['batch_size'],
            tau=ddpg_config['tau'],
            gamma=ddpg_config['gamma'],
            policy_kwargs=dict(net_arch=[128, 64]),
            verbose=0,
            seed=seed
        )
        
        # Train
        model.learn(
            total_timesteps=training_config['total_timesteps'],
            log_interval=training_config['log_interval']
        )
        
        # Save model
        model_path = os.path.join(self.models_dir, f'ddpg_seed_{seed}')
        model.save(model_path)
        
        return model
    
    def train_sac(self, seed: int = 0):
        """
        Train SAC agent.
        
        Args:
            seed: Random seed
        
        Returns:
            Trained model
        """
        print(f"\nTraining SAC (seed={seed})...")
        
        # Create environment
        env = self.create_env(self.train_data)
        
        # Get hyperparameters
        sac_config = self.config['models']['sac']
        training_config = self.config['training']
        
        # Create model
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=sac_config['learning_rate'],
            buffer_size=sac_config['buffer_size'],
            batch_size=sac_config['batch_size'],
            tau=sac_config['tau'],
            gamma=sac_config['gamma'],
            ent_coef=sac_config['ent_coef'],
            policy_kwargs=dict(net_arch=[128, 64]),
            verbose=0,
            seed=seed
        )
        
        # Train
        model.learn(
            total_timesteps=training_config['total_timesteps'],
            log_interval=training_config['log_interval']
        )
        
        # Save model
        model_path = os.path.join(self.models_dir, f'sac_seed_{seed}')
        model.save(model_path)
        
        return model
    
    def train_qr_ddpg(self, seed: int = 0):
        """
        Train QR-DDPG agent.
        
        Args:
            seed: Random seed
        
        Returns:
            Trained agent
        """
        print(f"\nTraining QR-DDPG (seed={seed})...")
        
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create environment
        env = self.create_env(self.train_data).envs[0]
        
        # Get hyperparameters
        qr_ddpg_config = self.config['models']['qr_ddpg']
        training_config = self.config['training']
        
        # Get state and action dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Create agent
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        agent = QRDDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=qr_ddpg_config['learning_rate_actor'],
            lr_critic=qr_ddpg_config['learning_rate_critic'],
            gamma=qr_ddpg_config['gamma'],
            tau=qr_ddpg_config['tau'],
            n_quantiles=qr_ddpg_config['n_quantiles'],
            buffer_size=qr_ddpg_config['buffer_size'],
            device=device
        )
        
        # Training loop
        total_timesteps = training_config['total_timesteps']
        batch_size = qr_ddpg_config['batch_size']
        
        state = env.reset()
        episode_reward = 0
        episode_count = 0
        
        for step in range(total_timesteps):
            # Select action
            action = agent.select_action(state, noise=0.1)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update agent
            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)
            
            episode_reward += reward
            state = next_state
            
            if done:
                episode_count += 1
                if episode_count % 10 == 0:
                    print(f"Episode {episode_count}, Reward: {episode_reward:.2f}")
                
                state = env.reset()
                episode_reward = 0
        
        # Save agent
        model_path = os.path.join(self.models_dir, f'qr_ddpg_seed_{seed}.pt')
        torch.save({
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
        }, model_path)
        
        return agent
    
    def evaluate_agent(self, model, agent_type: str = 'sb3'):
        """
        Evaluate trained agent on test data.
        
        Args:
            model: Trained model or agent
            agent_type: 'sb3' for Stable-Baselines3, 'custom' for custom agents
        
        Returns:
            Dictionary with evaluation results
        """
        # Create test environment
        env = self.create_env(self.test_data, is_training=False)
        
        if agent_type == 'sb3':
            env_unwrapped = env.envs[0]
        else:
            env_unwrapped = env
        
        # Run evaluation
        state = env_unwrapped.reset()
        done = False
        
        while not done:
            if agent_type == 'sb3':
                action, _ = model.predict(state, deterministic=True)
            else:
                action = model.select_action(state, noise=0.0)
            
            state, reward, done, info = env_unwrapped.step(action)
        
        # Get metrics
        metrics = env_unwrapped.get_portfolio_metrics()
        portfolio_df = env_unwrapped.save_portfolio_values()
        
        return metrics, portfolio_df
    
    def train_all_agents(self, n_seeds: int = None):
        """
        Train all agents with multiple seeds.
        
        Args:
            n_seeds: Number of random seeds (if None, use config value)
        """
        if n_seeds is None:
            n_seeds = self.config['training']['n_seeds']
        
        print("\n" + "="*50)
        print("STEP 2: Training DRL Agents")
        print("="*50)
        
        results = {
            'ppo': [],
            'ddpg': [],
            'sac': [],
            'qr_ddpg': []
        }
        
        for seed in range(n_seeds):
            print(f"\n--- Training with seed {seed} ---")
            
            # Train PPO
            ppo_model = self.train_ppo(seed)
            ppo_metrics, _ = self.evaluate_agent(ppo_model, 'sb3')
            results['ppo'].append(ppo_metrics)
            
            # Train DDPG
            ddpg_model = self.train_ddpg(seed)
            ddpg_metrics, _ = self.evaluate_agent(ddpg_model, 'sb3')
            results['ddpg'].append(ddpg_metrics)
            
            # Train SAC
            sac_model = self.train_sac(seed)
            sac_metrics, _ = self.evaluate_agent(sac_model, 'sb3')
            results['sac'].append(sac_metrics)
            
            # Train QR-DDPG
            qr_ddpg_agent = self.train_qr_ddpg(seed)
            qr_ddpg_metrics, _ = self.evaluate_agent(qr_ddpg_agent, 'custom')
            results['qr_ddpg'].append(qr_ddpg_metrics)
            
            print(f"\nSeed {seed} completed")
        
        # Save results
        self.save_results(results)
        
        return results
    
    def save_results(self, results: dict):
        """Save training results to CSV."""
        results_list = []
        
        for agent_name, metrics_list in results.items():
            for seed, metrics in enumerate(metrics_list):
                row = {'agent': agent_name, 'seed': seed}
                row.update(metrics)
                results_list.append(row)
        
        results_df = pd.DataFrame(results_list)
        results_path = os.path.join(self.results_dir, 'training_results.csv')
        results_df.to_csv(results_path, index=False)
        
        print(f"\nResults saved to: {results_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        
        summary = results_df.groupby('agent').agg({
            'annual_return': ['mean', 'std'],
            'sharpe_ratio': ['mean', 'std'],
            'max_drawdown': ['mean', 'std'],
            'sortino_ratio': ['mean', 'std']
        })
        
        print(summary)


def main():
    """Main training function."""
    trainer = TrainDRLAgents()
    
    # Prepare data
    trainer.prepare_data()
    
    # Train all agents (use 2 seeds for quick demo, set to 10 for full training)
    results = trainer.train_all_agents(n_seeds=2)
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
