"""PPO (Proximal Policy Optimization) Algorithm"""
import os
import random
import sys
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import make_env_discrete, make_env_continuous, PPODiscreteAgent, PPOContinuousAgent


class PPO:
    """PPO Algorithm Trainer"""
    
    def __init__(
        self,
        env_id: str,
        seed: int = 1,
        total_timesteps: int = 500000,
        learning_rate: float = 2.5e-4,
        num_envs: int = 4,
        num_steps: int = 128,
        anneal_lr: bool = True,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        num_minibatches: int = 4,
        update_epochs: int = 4,
        norm_adv: bool = True,
        clip_coef: float = 0.2,
        clip_vloss: bool = True,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        convergence_threshold_ratio: float = 0.99,
        cuda: bool = True,
        torch_deterministic: bool = True,
        capture_video: bool = False,
        track: bool = False,
        wandb_project_name: str = "cleanRL",
        wandb_entity: Optional[str] = None,
        exp_name: str = "ppo",
    ):
        self.env_id = env_id
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.anneal_lr = anneal_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.convergence_threshold_ratio = convergence_threshold_ratio
        self.cuda = cuda
        self.torch_deterministic = torch_deterministic
        self.capture_video = capture_video
        self.track = track
        self.wandb_project_name = wandb_project_name
        self.wandb_entity = wandb_entity
        self.exp_name = exp_name
        
        # Compute runtime parameters
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_iterations = self.total_timesteps // self.batch_size
        
        # Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cuda else "cpu")
        self.run_name = f"{self.env_id}__{self.exp_name}__{self.seed}__{int(time.time())}"
        
        # Initialize wandb if tracking
        if self.track:
            import wandb
            wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity,
                sync_tensorboard=True,
                config=self.get_config(),
                name=self.run_name,
                monitor_gym=True,
                save_code=True,
            )
        
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in self.get_config().items()])),
        )
        
        # Set seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic
        
        # GPU optimization settings (CleanRL style)
        # Note: benchmark=True conflicts with deterministic=True, so only enable when not deterministic
        if torch.cuda.is_available() and self.cuda:
            torch.backends.cudnn.benchmark = not self.torch_deterministic
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Create environment (detect action space type)
        self.envs = None
        self.agent = None
        self.optimizer = None
        self.is_discrete = None
        
        self.all_episode_rewards = []
        self.convergence_episode = None
        self.max_performance_episode = None
        self.eval_window_size = 100
        
    def get_config(self):
        """Get configuration dictionary"""
        return {
            'env_id': self.env_id,
            'seed': self.seed,
            'total_timesteps': self.total_timesteps,
            'learning_rate': self.learning_rate,
            'num_envs': self.num_envs,
            'num_steps': self.num_steps,
            'anneal_lr': self.anneal_lr,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'num_minibatches': self.num_minibatches,
            'update_epochs': self.update_epochs,
            'norm_adv': self.norm_adv,
            'clip_coef': self.clip_coef,
            'clip_vloss': self.clip_vloss,
            'ent_coef': self.ent_coef,
            'vf_coef': self.vf_coef,
            'max_grad_norm': self.max_grad_norm,
            'target_kl': self.target_kl,
        }
    
    def _setup_env(self):
        """Setup environment and agent"""
        self.envs = gym.vector.SyncVectorEnv(
            [make_env_discrete(self.env_id, i, self.capture_video, self.run_name, self.seed) 
             for i in range(self.num_envs)]
        )
        
        self.is_discrete = isinstance(self.envs.single_action_space, gym.spaces.Discrete)
        
        if self.is_discrete:
            self.agent = PPODiscreteAgent(self.envs).to(self.device)
        else:
            self.envs.close()
            self.envs = gym.vector.SyncVectorEnv(
                [make_env_continuous(self.env_id, i, self.capture_video, self.run_name, self.gamma, self.seed) 
                 for i in range(self.num_envs)]
            )
            self.agent = PPOContinuousAgent(self.envs).to(self.device)
        
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)
    
    def train(self):
        """Train the agent"""
        if self.envs is None:
            self._setup_env()
        
        # Storage setup
        obs = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_observation_space.shape).to(self.device)
        actions = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset(seed=self.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        
        # Progress bar
        pbar = tqdm(range(1, self.num_iterations + 1), desc="Training", unit="iter")
        
        for iteration in pbar:
            # Anneal learning rate
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow
            
            # Collect rollouts
            for step in range(0, self.num_steps):
                global_step += self.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                
                # Execute step
                next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)
                
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            episode_reward = float(info["episode"]["r"])
                            self.writer.add_scalar("charts/episodic_return", episode_reward, global_step)
                            self.writer.add_scalar("charts/episodic_return_iteration", episode_reward, iteration)
                            self.all_episode_rewards.append(episode_reward)
                elif "episode" in infos:
                    episode_info = infos["episode"]
                    if "_r" in episode_info:
                        for i in range(len(episode_info["_r"])):
                            if episode_info["_r"][i]:
                                episode_reward = float(episode_info["r"][i])
                                self.writer.add_scalar("charts/episodic_return", episode_reward, global_step)
                                self.writer.add_scalar("charts/episodic_return_iteration", episode_reward, iteration)
                                self.all_episode_rewards.append(episode_reward)
            
            # Compute advantages
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            
            # Flatten batch
            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            
            # Update policy
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]
                    
                    if self.is_discrete:
                        _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                            b_obs[mb_inds], b_actions.long()[mb_inds]
                        )
                    else:
                        _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                            b_obs[mb_inds], b_actions[mb_inds]
                        )
                    
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
                    
                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    
                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    
                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                if self.target_kl is not None and approx_kl > self.target_kl:
                    break
            
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            
            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
            
            self.writer.add_scalar("losses/value_loss_iteration", v_loss.item(), iteration)
            
            if len(self.all_episode_rewards) >= self.eval_window_size:
                if self.max_performance_episode is None and len(self.all_episode_rewards) >= 10:
                    max_performance = max(self.all_episode_rewards)
                    for i, reward in enumerate(self.all_episode_rewards):
                        if reward >= max_performance * self.convergence_threshold_ratio:
                            self.max_performance_episode = i
                            break
                
                if self.convergence_episode is None and len(self.all_episode_rewards) >= 50:
                    max_performance = max(self.all_episode_rewards)
                    convergence_threshold = max_performance * self.convergence_threshold_ratio
                    window_size = 10
                    for i in range(len(self.all_episode_rewards) - window_size + 1):
                        window_rewards = self.all_episode_rewards[i:i+window_size]
                        window_avg = np.mean(window_rewards)
                        if window_avg >= convergence_threshold:
                            self.convergence_episode = i
                            threshold_pct = int(self.convergence_threshold_ratio * 100)
                            print(f"\n[Convergence] Episode {i}: Moving avg {window_avg:.2f} >= Threshold {convergence_threshold:.2f} ({threshold_pct}% of Max: {max_performance:.2f})")
                            break
                
                if self.max_performance_episode is not None and self.max_performance_episode < len(self.all_episode_rewards):
                    eval_rewards = self.all_episode_rewards[self.max_performance_episode:]
                elif self.convergence_episode is not None and self.convergence_episode < len(self.all_episode_rewards):
                    eval_rewards = self.all_episode_rewards[self.convergence_episode:]
                else:
                    eval_rewards = self.all_episode_rewards[-self.eval_window_size:]
                
                return_avg = np.mean(eval_rewards)
                reward_variance = np.std(eval_rewards, ddof=1)
                
                self.writer.add_scalar("paper_metrics/return", return_avg, global_step)
                self.writer.add_scalar("paper_metrics/reward_variance", reward_variance, global_step)
                
                if self.convergence_episode is not None:
                    self.writer.add_scalar("paper_metrics/convergence_steps", self.convergence_episode, global_step)
            
            pbar.set_postfix({"clip_coef": f"{self.clip_coef:.3f}"})
        
        pbar.close()
        
        if len(self.all_episode_rewards) > 0:
            if self.max_performance_episode is not None and self.max_performance_episode < len(self.all_episode_rewards):
                final_rewards = self.all_episode_rewards[self.max_performance_episode:]
                eval_note = f"Post-max-performance ({len(final_rewards)} episodes)"
            elif self.convergence_episode is not None and self.convergence_episode < len(self.all_episode_rewards):
                eval_rewards = self.all_episode_rewards[self.convergence_episode:]
                eval_window = min(self.eval_window_size, len(eval_rewards))
                final_rewards = eval_rewards[-eval_window:]
                eval_note = f"Post-convergence ({eval_window} episodes)"
            else:
                eval_window = min(self.eval_window_size, len(self.all_episode_rewards))
                final_rewards = self.all_episode_rewards[-eval_window:]
                eval_note = f"Last {eval_window} episodes"
            
            final_return = np.mean(final_rewards)
            final_variance = np.std(final_rewards, ddof=1)
            
            self.writer.add_scalar("paper_metrics/final_return", final_return, global_step)
            self.writer.add_scalar("paper_metrics/final_reward_variance", final_variance, global_step)
            
            if self.convergence_episode is not None:
                self.writer.add_scalar("paper_metrics/final_convergence_steps", self.convergence_episode, global_step)
            
            print(f"\n{'='*60}")
            print("Final Paper Reproduction Metrics:")
            print(f"{'='*60}")
            print(f"Return (Avg): {final_return:.2f}")
            print(f"Reward Variance: {final_variance:.2f}")
            if self.convergence_episode is not None:
                max_performance = max(self.all_episode_rewards)
                convergence_threshold = max_performance * self.convergence_threshold_ratio
                threshold_pct = int(self.convergence_threshold_ratio * 100)
                print(f"Convergence Steps (Episodes): {self.convergence_episode}")
                print(f"Convergence Threshold: {convergence_threshold:.2f} ({threshold_pct}% of max: {max_performance:.2f})")
            else:
                print(f"Convergence Steps: Not converged")
            print(f"Evaluation Window: {eval_note}")
            print(f"{'='*60}")
        
        self.envs.close()
        self.writer.close()
        
        return self.agent

