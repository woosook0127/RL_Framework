"""Simple PPO-BR: Minimal modification to PPO (testing paper's "5 lines" claim)"""
import os
import random
import sys
import time
from collections import deque
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


class Simple_PPO_BR:
    """
    Simple PPO-BR: Minimal modification to PPO to test paper's "5 lines" claim.
    
    This is a simplified version that adds only the essential adaptive clipping logic
    to the PPO algorithm, following the paper's formula as closely as possible.
    """
    
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
        epsilon_0: float = 0.2,  # Base clipping threshold ε_0
        clip_vloss: bool = True,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        lambda_1: float = 0.5,  # Scaling for entropy expansion λ_1
        lambda_2: float = 0.3,  # Scaling for reward contraction λ_2
        reward_window_size: int = 10,  # Window for reward progression
        psi_method: str = 'tanh',  # Method for ψ(ΔR_t): 'tanh' (ppo_br), 'inverse', 'exp', 'tanh_scaled', 'smooth'
        delta_r_method: str = 'half_mean',  # Method for ΔR_t: 'half_mean' (ppo_br), 'linear', 'exponential', 'simple_mean'
        convergence_threshold_ratio: float = 0.99,  # Convergence threshold as ratio of max performance
        cuda: bool = True,
        torch_deterministic: bool = True,
        capture_video: bool = False,
        track: bool = False,
        wandb_project_name: str = "cleanRL",
        wandb_entity: Optional[str] = None,
        exp_name: str = "simple_ppo_br",
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
        self.epsilon_0 = epsilon_0
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.reward_window_size = reward_window_size
        self.psi_method = psi_method
        self.delta_r_method = delta_r_method
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
        
        # GPU optimization settings
        if torch.cuda.is_available() and self.cuda:
            torch.backends.cudnn.benchmark = not self.torch_deterministic
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.envs = None
        self.agent = None
        self.optimizer = None
        self.is_discrete = None
        
        # PPO-BR additions (minimal)
        self.recent_rewards = deque(maxlen=self.reward_window_size)
        self.H_max = None  # Will be set in _setup_env
        
        self.all_episode_rewards = []
        self.convergence_episode = None
        self.max_performance_episode = None
        self.eval_window_size = 100
        # convergence_threshold_ratio is already set from parameter above
    
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
            'epsilon_0': self.epsilon_0,
            'clip_vloss': self.clip_vloss,
            'ent_coef': self.ent_coef,
            'vf_coef': self.vf_coef,
            'max_grad_norm': self.max_grad_norm,
            'target_kl': self.target_kl,
            'lambda_1': self.lambda_1,
            'lambda_2': self.lambda_2,
            'reward_window_size': self.reward_window_size,
            'psi_method': self.psi_method,
            'delta_r_method': self.delta_r_method,
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
            self.H_max = np.log(self.envs.single_action_space.n)
        else:
            self.envs.close()
            self.envs = gym.vector.SyncVectorEnv(
                [make_env_continuous(self.env_id, i, self.capture_video, self.run_name, self.gamma, self.seed) 
                 for i in range(self.num_envs)]
            )
            self.agent = PPOContinuousAgent(self.envs).to(self.device)
            action_dim = np.prod(self.envs.single_action_space.shape)
            max_entropy_per_dim = 0.5 * np.log(2 * np.pi * np.e * 1.0)
            self.H_max = action_dim * max_entropy_per_dim
        
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)
    
    def train(self):
        """Train the agent - PPO with minimal PPO-BR modifications"""
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
                
                # PPO-BR addition: Track episode rewards
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            episode_reward = float(info["episode"]["r"])
                            self.writer.add_scalar("charts/episodic_return", episode_reward, global_step)
                            self.recent_rewards.append(episode_reward)
                            self.all_episode_rewards.append(episode_reward)
                elif "episode" in infos:
                    episode_info = infos["episode"]
                    if "_r" in episode_info:
                        for i in range(len(episode_info["_r"])):
                            if episode_info["_r"][i]:
                                episode_reward = float(episode_info["r"][i])
                                self.writer.add_scalar("charts/episodic_return", episode_reward, global_step)
                                self.recent_rewards.append(episode_reward)
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
            
            # ============================================================
            # PPO-BR: Compute adaptive clipping threshold (matching ppo_br.py)
            # ============================================================
            with torch.no_grad():
                _, _, entropy_batch, _ = self.agent.get_action_and_value(b_obs)
                H_t = entropy_batch.mean().item()
            
            # Normalization function φ: maps H_t to [0,1]
            phi_H_t = np.clip(H_t / (self.H_max + 1e-8), 0.0, 1.0)
            
            # Compute reward progression ΔR_t (multiple methods)
            Delta_R_t = 0.0
            # Use reward_window_size (옵션 1, 2와 동일한 조건)
            if len(self.recent_rewards) >= self.reward_window_size:
                reward_array = np.array(self.recent_rewards)
                
                if self.delta_r_method == 'half_mean':
                    # Method 1: Same as ppo_br.py (default) - half mean difference
                    mid = len(reward_array) // 2
                    R_first = np.mean(reward_array[:mid])
                    R_second = np.mean(reward_array[mid:])
                    if abs(R_first) > 1e-8:
                        Delta_R_t = (R_second - R_first) / (abs(R_first) + 1e-8)
                    else:
                        Delta_R_t = 0.0
                elif self.delta_r_method == 'linear':
                    # Method 2: Linear regression slope (옵션 1 - ISSUES_AND_RECOMMENDATIONS.md)
                    x = np.arange(len(reward_array))
                    slope = np.polyfit(x, reward_array, 1)[0]
                    reward_std = np.std(reward_array)
                    if reward_std > 1e-8:
                        Delta_R_t = slope / (reward_std + 1e-8)
                    else:
                        Delta_R_t = 0.0
                elif self.delta_r_method == 'simple_mean':
                    # Method 5: Simple mean difference (옵션 2 - ISSUES_AND_RECOMMENDATIONS.md)
                    # 단순 평균 차이, 정규화는 선택적
                    mid = len(reward_array) // 2
                    Delta_R_t = np.mean(reward_array[mid:]) - np.mean(reward_array[:mid])
                    # 정규화는 선택적 (사용자 코드와 동일)
                    if abs(np.mean(reward_array[:mid])) > 1e-8:
                        Delta_R_t = Delta_R_t / (abs(np.mean(reward_array[:mid])) + 1e-8)
                elif self.delta_r_method == 'exponential':
                    # Method 3: Exponential weighted difference (recent rewards weighted more)
                    weights = np.exp(np.linspace(-1, 0, len(reward_array)))
                    weights = weights / weights.sum()
                    weighted_mean = np.average(reward_array, weights=weights)
                    unweighted_mean = np.mean(reward_array)
                    if abs(unweighted_mean) > 1e-8:
                        Delta_R_t = (weighted_mean - unweighted_mean) / (abs(unweighted_mean) + 1e-8)
                    else:
                        Delta_R_t = 0.0
                elif self.delta_r_method == 'recent_vs_old':
                    # Method 4: Recent third vs old third (more aggressive)
                    third = len(reward_array) // 3
                    R_old = np.mean(reward_array[:third])
                    R_recent = np.mean(reward_array[-third:])
                    if abs(R_old) > 1e-8:
                        Delta_R_t = (R_recent - R_old) / (abs(R_old) + 1e-8)
                    else:
                        Delta_R_t = 0.0
                else:
                    # Default to half_mean
                    mid = len(reward_array) // 2
                    R_first = np.mean(reward_array[:mid])
                    R_second = np.mean(reward_array[mid:])
                    if abs(R_first) > 1e-8:
                        Delta_R_t = (R_second - R_first) / (abs(R_first) + 1e-8)
                    else:
                        Delta_R_t = 0.0
            
            # Normalization function ψ: maps ΔR_t to [0,1]
            # High when plateauing (convergence), low when changing (exploration)
            abs_Delta_R_t = abs(Delta_R_t)
            if self.psi_method == 'tanh':
                # Method 1: Same as ppo_br.py (default)
                psi_Delta_R_t = 1.0 - np.tanh(abs_Delta_R_t)
            elif self.psi_method == 'inverse':
                # Method 2: Inverse function (simpler)
                psi_Delta_R_t = 1.0 / (1.0 + abs_Delta_R_t)
            elif self.psi_method == 'exp':
                # Method 3: Exponential decay
                psi_Delta_R_t = np.exp(-abs_Delta_R_t)
            elif self.psi_method == 'tanh_scaled':
                # Method 4: Scaled tanh (more sensitive)
                psi_Delta_R_t = 1.0 - np.tanh(abs_Delta_R_t * 2.0)
            elif self.psi_method == 'smooth':
                # Method 5: Smooth transition (sigmoid-like)
                psi_Delta_R_t = 1.0 / (1.0 + abs_Delta_R_t ** 2)
            elif self.psi_method == 'adaptive':
                # Method 6: Adaptive based on reward scale
                reward_scale = np.std(self.recent_rewards) if len(self.recent_rewards) > 1 else 1.0
                normalized_delta = abs_Delta_R_t / (reward_scale + 1e-8)
                psi_Delta_R_t = 1.0 - np.tanh(normalized_delta)
            else:
                # Default to tanh
                psi_Delta_R_t = 1.0 - np.tanh(abs_Delta_R_t)
            
            # Apply paper formula: ε_t = ε_0 * (1 + λ_1 * tanh(φ(H_t)) - λ_2 * tanh(ψ(ΔR_t)))
            epsilon_t = self.epsilon_0 * (
                1.0 + 
                self.lambda_1 * np.tanh(phi_H_t) - 
                self.lambda_2 * np.tanh(psi_Delta_R_t)
            )
            
            # Enforce bounds from Lemma 1: ε_t ∈ [ε_0(1-λ_2), ε_0(1+λ_1)]
            epsilon_t = np.clip(epsilon_t, self.epsilon_0 * (1 - self.lambda_2), self.epsilon_0 * (1 + self.lambda_1))
            # ============================================================
            
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
                        clipfracs += [((ratio - 1.0).abs() > epsilon_t).float().mean().item()]
                    
                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    
                    # Policy loss with adaptive epsilon_t
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - epsilon_t, 1 + epsilon_t)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # Value loss (use epsilon_0 for value clipping, matching ppo_br's value_clip_coef default)
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.epsilon_0,  # Use epsilon_0 (same as ppo_br's value_clip_coef default)
                            self.epsilon_0,
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
            
            # Essential logs for PPO-BR tuning
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            # PPO-BR core metrics
            self.writer.add_scalar("ppo_br/epsilon_t", epsilon_t, global_step)
            self.writer.add_scalar("ppo_br/H_t", H_t, global_step)
            self.writer.add_scalar("ppo_br/Delta_R_t", Delta_R_t, global_step)
            
            # Disabled logs (uncomment if needed for debugging)
            # y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            # var_y = np.var(y_true)
            # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            # self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            # self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            # self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            # self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            # self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
            
            # Convergence tracking (same as ppo_br)
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
                
                # Essential paper metrics (logged periodically for monitoring)
                self.writer.add_scalar("paper_metrics/return", return_avg, global_step)
                self.writer.add_scalar("paper_metrics/reward_variance", reward_variance, global_step)
                
                # Disabled (uncomment if needed)
                # if self.convergence_episode is not None:
                #     self.writer.add_scalar("paper_metrics/convergence_steps", self.convergence_episode, global_step)
            
            pbar.set_postfix({"eps_t": f"{epsilon_t:.3f}", "H_t": f"{H_t:.2f}", "ΔR_t": f"{Delta_R_t:.3f}"})
        
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
            
            # Final metrics (essential for evaluation)
            self.writer.add_scalar("paper_metrics/final_return", final_return, global_step)
            self.writer.add_scalar("paper_metrics/final_reward_variance", final_variance, global_step)
            
            # Disabled (uncomment if needed)
            # if self.convergence_episode is not None:
            #     self.writer.add_scalar("paper_metrics/final_convergence_steps", self.convergence_episode, global_step)
            
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

