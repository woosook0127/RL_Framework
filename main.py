#!/usr/bin/env python3
"""RL Framework 통합 실행 인터페이스"""
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms import PPO, PPO_BR, Simple_PPO_BR


DEFAULT_COMMON = {
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'num_minibatches': 4,
    'update_epochs': 4,
    'norm_adv': True,
    'anneal_lr': True,
    'clip_vloss': True,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'target_kl': None,
    'cuda': True,
    'torch_deterministic': True,
}

PPO_DEFAULT = {
    **DEFAULT_COMMON,
    'learning_rate': 3e-4,
    'clip_coef': 0.2,
}

PPO_BR_DEFAULT = {
    **DEFAULT_COMMON,
    'learning_rate': 3e-4,
    'epsilon_0': 0.2,  # Base clipping threshold ε_0 (paper notation)
    'value_clip_coef': 0.2,
    'lambda_1': 0.5,  # Scaling hyperparameter for entropy expansion λ_1 (paper notation)
    'lambda_2': 0.3,  # Scaling hyperparameter for reward contraction λ_2 (paper notation)
    'reward_window_size': 10,
}

SIMPLE_PPO_BR_DEFAULT = {
    **DEFAULT_COMMON,
    'learning_rate': 3e-4,
    'epsilon_0': 0.2,  # Base clipping threshold ε_0 (paper notation)
    'lambda_1': 0.5,  # Scaling hyperparameter for entropy expansion λ_1 (paper notation)
    'lambda_2': 0.3,  # Scaling hyperparameter for reward contraction λ_2 (paper notation)
    'reward_window_size': 20,  # Match ppo_br default (was 10)
    'psi_method': 'smooth',  # Best: 'smooth' (exponential + smooth is best combination)
    'delta_r_method': 'exponential',  # Best: 'exponential' (exponential + smooth is best combination)
}

ENV_CONFIGS = {
    'CartPole': {
        'total_timesteps': 500000,
        'num_envs': 4,
        'num_steps': 128,
        'ent_coef': 0.01,
        'convergence_threshold_ratio': 0.99,
        'reward_window_size': 10,  # 빠른 수렴, 짧은 에피소드 → 작은 k
    },
    'LunarLander': {  # Best version
        'total_timesteps': 1000000,
        'num_envs': 4,
        'num_steps': 128,
        'epsilon_0': 0.3,
        'ent_coef': 0.02,  # 0.01 -> 0.02: Increase entropy bonus for exploration
        'lambda_1': 0.8,  # 0.5 -> 0.8: Strong expansion for better exploration
        'lambda_2': 0.3,
        'reward_window_size': 15,  # 20 -> 15: Smaller window for faster adaptation
        'convergence_threshold_ratio': 0.5,
    },
    'Hopper': { # Best version. PPO 와 비슷슷
        'total_timesteps': 3000000,
        'num_envs': 1,
        'num_steps': 2048,
        'epsilon_0': 0.23,  # Gray(0.25)와 현재(0.2)의 균형: 안정성 + 높은 성능
        'ent_coef': 0.0005,  # 매우 작은 entropy bonus: 초반 exploration만 도움
        'reward_window_size': 22,  # Gray(30)와 현재(20)의 중간: 빠른 적응 + 안정성
        'lambda_1': 0.65,  # Gray(0.7)와 현재(0.5)의 중간: 강한 exploration으로 높은 최종 성능
        'lambda_2': 0.2,  # Gray와 동일: Expansion 여유 확보
    },
    # 'Hopper': { # Best version. PPO 와 비슷슷
    #     'total_timesteps': 3000000,
    #     'num_envs': 1,
    #     'num_steps': 2048,
    #     'epsilon_0': 0.23,  # Gray(0.25)와 현재(0.2)의 균형: 안정성 + 높은 성능
    #     'ent_coef': 0.0005,  # 매우 작은 entropy bonus: 초반 exploration만 도움
    #     'reward_window_size': 22,  # Gray(30)와 현재(20)의 중간: 빠른 적응 + 안정성
    #     'lambda_1': 0.65,  # Gray(0.7)와 현재(0.5)의 중간: 강한 exploration으로 높은 최종 성능
    #     'lambda_2': 0.2,  # Gray와 동일: Expansion 여유 확보
    # },
    'HalfCheetah': {
        'total_timesteps': 3000000,
        'num_envs': 1,
        'num_steps': 2048,
        'ent_coef': 0.0,
        # Critical fix: Initial return too low → early convergence to poor performance
        # Strategy: (1) Better early exploration, (2) Prevent premature convergence, (3) Enable long-term learning
        'update_epochs': 4,  # Standard epochs
        'vf_coef': 0.5,  # Standard value function coefficient
        'max_grad_norm': 0.5,  # Standard gradient clipping
    },
    'Walker2d': {
        'total_timesteps': 3000000,
        'num_envs': 1,
        'num_steps': 2048,
        'ent_coef': 0.0,
        # PPO-BR paper principles: dual-signal adaptation
        # (1) Entropy-driven expansion: promotes exploration in high-uncertainty states
        # (2) Reward-guided contraction: enforces stability during convergence
        'learning_rate': 3e-4,  # Standard learning rate
        'epsilon_0': 0.2,  # Base clipping threshold (paper default)
        'lambda_1': 0.65,  # Enhanced entropy expansion for faster early exploration
        'lambda_2': 0.28,  # Strong reward contraction for stable convergence (paper emphasizes this)
        'reward_window_size': 25,  # Optimal window for tracking reward progression (ΔR_t)
        'update_epochs': 4,  # Standard epochs
    },
    'Humanoid': {
        'total_timesteps': 10000000,
        'num_envs': 1,
        'num_steps': 2048,
        'ent_coef': 0.0,
        'reward_window_size': 40,  # 매우 느린 수렴, 매우 높은 변동성 → 매우 큰 k
    },
}

ALL_ENVS = ['CartPole-v0', 'LunarLander-v3', 'Hopper-v4', 'HalfCheetah-v5', 'Walker2d-v5', 'Humanoid-v5']


def _get_env_name(env_id):
    """Extract environment name from env_id (e.g., 'Hopper-v5' -> 'Hopper')"""
    if '-' in env_id:
        return env_id.split('-')[0]
    return env_id


def run_experiment(env_id, algorithm='ppo_br', **kwargs):
    """단일 환경에서 실험 실행"""
    if algorithm.lower() == 'ppo':
        default_config = PPO_DEFAULT.copy()
    elif algorithm.lower() == 'ppo_br':
        default_config = PPO_BR_DEFAULT.copy()
    elif algorithm.lower() == 'simple_ppo_br':
        default_config = SIMPLE_PPO_BR_DEFAULT.copy()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    env_name = _get_env_name(env_id)
    env_config = ENV_CONFIGS.get(env_name, {}).copy()
    
    config = {**default_config, **env_config, **kwargs}
    config['env_id'] = env_id
    
    print(f"\n{'='*60}")
    print(f"Running {algorithm.upper()} on {env_id}")
    print(f"{'='*60}")
    
    if algorithm.lower() == 'ppo':
        ppo_config = {k: v for k, v in config.items() 
                     if k not in ['lambda_1', 'lambda_2', 'lambda1', 'lambda2', 'reward_window_size', 
                                 'min_clip_coef', 'max_clip_coef', 'base_clip_coef', 'epsilon_0', 'value_clip_coef']}
        trainer = PPO(**ppo_config)
    elif algorithm.lower() == 'ppo_br':
        ppo_br_config = config.copy()
        # Handle backward compatibility: convert old names to new paper notation
        if 'clip_coef' in ppo_br_config:
            if 'epsilon_0' not in ppo_br_config and 'base_clip_coef' not in ppo_br_config:
                ppo_br_config['epsilon_0'] = ppo_br_config.pop('clip_coef')
            else:
                ppo_br_config.pop('clip_coef')
        if 'base_clip_coef' in ppo_br_config and 'epsilon_0' not in ppo_br_config:
            ppo_br_config['epsilon_0'] = ppo_br_config.pop('base_clip_coef')
        if 'lambda1' in ppo_br_config and 'lambda_1' not in ppo_br_config:
            ppo_br_config['lambda_1'] = ppo_br_config.pop('lambda1')
        if 'lambda2' in ppo_br_config and 'lambda_2' not in ppo_br_config:
            ppo_br_config['lambda_2'] = ppo_br_config.pop('lambda2')
        trainer = PPO_BR(**ppo_br_config)
    elif algorithm.lower() == 'simple_ppo_br':
        simple_ppo_br_config = config.copy()
        # Handle backward compatibility: convert old names to new paper notation
        if 'clip_coef' in simple_ppo_br_config:
            if 'epsilon_0' not in simple_ppo_br_config and 'base_clip_coef' not in simple_ppo_br_config:
                simple_ppo_br_config['epsilon_0'] = simple_ppo_br_config.pop('clip_coef')
            else:
                simple_ppo_br_config.pop('clip_coef')
        if 'base_clip_coef' in simple_ppo_br_config and 'epsilon_0' not in simple_ppo_br_config:
            simple_ppo_br_config['epsilon_0'] = simple_ppo_br_config.pop('base_clip_coef')
        if 'lambda1' in simple_ppo_br_config and 'lambda_1' not in simple_ppo_br_config:
            simple_ppo_br_config['lambda_1'] = simple_ppo_br_config.pop('lambda1')
        if 'lambda2' in simple_ppo_br_config and 'lambda_2' not in simple_ppo_br_config:
            simple_ppo_br_config['lambda_2'] = simple_ppo_br_config.pop('lambda2')
        # Remove PPO_BR specific params that Simple_PPO_BR doesn't use
        simple_ppo_br_config.pop('value_clip_coef', None)
        # Keep convergence_threshold_ratio (Simple_PPO_BR now supports it)
        trainer = Simple_PPO_BR(**simple_ppo_br_config)
    
    agent = trainer.train()
    
    print(f"\n✓ Training completed!")
    return agent


def run_all_experiments(algorithm='ppo_br', **kwargs):
    """모든 환경에서 실험 실행"""
    print(f"\n{'='*60}")
    print(f"Running {algorithm.upper()} on all environments")
    print(f"Environments: {', '.join(ALL_ENVS)}")
    print(f"{'='*60}\n")
    
    results = {}
    for env_id in ALL_ENVS:
        try:
            agent = run_experiment(env_id, algorithm=algorithm, **kwargs)
            results[env_id] = {'success': True, 'agent': agent}
        except Exception as e:
            print(f"\n✗ Failed on {env_id}: {e}")
            results[env_id] = {'success': False, 'error': str(e)}
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for env_id, result in results.items():
        status = "✓" if result.get('success') else "✗"
        print(f"{status} {env_id}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='RL Framework: PPO and PPO-BR unified interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # PPO-BR on CartPole
  python main.py --algorithm ppo_br --env CartPole-v1 --seed 42

  # Simple PPO-BR on LunarLander (minimal implementation)
  python main.py --algorithm simple_ppo_br --env LunarLander-v3 --seed 42

  # PPO on HalfCheetah
  python main.py --algorithm ppo --env HalfCheetah-v4 --seed 42

  # PPO-BR on all environments
  python main.py --algorithm ppo_br --env all

  # With custom parameters
  python main.py --algorithm ppo_br --env CartPole-v1 --total-timesteps 1000000 --lambda1 0.7
        """
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        default='ppo_br',
        choices=['ppo', 'ppo_br', 'simple_ppo_br'],
        help='Algorithm to use (default: ppo_br). simple_ppo_br is a minimal implementation for testing.'
    )
    
    parser.add_argument(
        '--env',
        type=str,
        default='CartPole-v1',
        help='Environment ID (default: CartPole-v1). Use "all" to run on all environments.'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42). Use --num-seeds to run multiple seeds.'
    )
    
    parser.add_argument(
        '--num-seeds',
        type=int,
        default=1,
        help='Number of random seeds to run (default: 1). If > 1, runs experiments with seeds from --start-seed to --start-seed + --num-seeds - 1'
    )
    
    parser.add_argument(
        '--start-seed',
        type=int,
        default=1,
        help='Starting seed number (default: 1). Used with --num-seeds'
    )
    
    # Common hyperparameters
    parser.add_argument('--total-timesteps', type=int, default=None, help='Total timesteps')
    parser.add_argument('--learning-rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--num-envs', type=int, default=None, help='Number of parallel environments')
    parser.add_argument('--num-steps', type=int, default=None, help='Steps per rollout')
    parser.add_argument('--gamma', type=float, default=None, help='Discount factor (default: 0.99)')
    parser.add_argument('--gae-lambda', type=float, default=None, help='GAE lambda (default: 0.95)')
    parser.add_argument('--update-epochs', type=int, default=None, help='Update epochs (default: 4)')
    parser.add_argument('--num-minibatches', type=int, default=None, help='Number of minibatches (default: 4)')
    parser.add_argument('--norm-adv', type=lambda x: x.lower() in ['true', '1', 'yes'], default=None, metavar='BOOL', help='Normalize advantages (default: True)')
    parser.add_argument('--anneal-lr', type=lambda x: x.lower() in ['true', '1', 'yes'], default=None, metavar='BOOL', help='Anneal learning rate (default: True)')
    parser.add_argument('--clip-vloss', type=lambda x: x.lower() in ['true', '1', 'yes'], default=None, metavar='BOOL', help='Clip value loss (default: True)')
    parser.add_argument('--clip-coef', type=float, default=None, help='Clipping coefficient (PPO, default: 0.2)')
    parser.add_argument('--epsilon-0', type=float, default=None, help='Base clipping threshold ε_0 (PPO-BR, paper notation, default: 0.2)')
    parser.add_argument('--base-clip-coef', type=float, default=None, help='[Deprecated] Use --epsilon-0 instead. Base clipping coefficient (PPO-BR, default: 0.2)')
    parser.add_argument('--value-clip-coef', type=float, default=None, help='Value clipping coefficient (PPO-BR, default: 0.2)')
    parser.add_argument('--ent-coef', type=float, default=None, help='Entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=None, help='Value function coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=None, help='Max gradient norm (default: 0.5)')
    parser.add_argument('--target-kl', type=float, default=None, help='Target KL divergence (default: None)')
    
    # PPO-BR specific (paper notation)
    parser.add_argument('--lambda-1', type=float, default=None, help='PPO-BR λ_1: scaling hyperparameter for entropy expansion (paper notation, default: 0.5)')
    parser.add_argument('--lambda-2', type=float, default=None, help='PPO-BR λ_2: scaling hyperparameter for reward contraction (paper notation, default: 0.3)')
    parser.add_argument('--lambda1', type=float, default=None, help='[Deprecated] Use --lambda-1 instead. PPO-BR lambda1 (entropy expansion, default: 0.5)')
    parser.add_argument('--lambda2', type=float, default=None, help='[Deprecated] Use --lambda-2 instead. PPO-BR lambda2 (reward contraction, default: 0.3)')
    parser.add_argument('--reward-window-size', type=int, default=None, help='Reward window size for PPO-BR (default: 10)')
    parser.add_argument('--psi-method', type=str, default=None, choices=['tanh', 'inverse', 'exp', 'tanh_scaled', 'smooth', 'adaptive'], help='Method for ψ(ΔR_t) in Simple PPO-BR: tanh (ppo_br), inverse, exp, tanh_scaled, smooth, adaptive (default: tanh)')
    parser.add_argument('--delta-r-method', type=str, default=None, choices=['half_mean', 'linear', 'exponential', 'recent_vs_old', 'simple_mean'], help='Method for ΔR_t in Simple PPO-BR: half_mean (ppo_br), linear (옵션1), exponential, recent_vs_old, simple_mean (옵션2) (default: exponential)')
    
    # Other options
    parser.add_argument('--cuda', type=lambda x: x.lower() in ['true', '1', 'yes'], default=None, metavar='BOOL', help='Use CUDA (default: True)')
    parser.add_argument('--torch-deterministic', type=lambda x: x.lower() in ['true', '1', 'yes'], default=None, metavar='BOOL', help='Use deterministic torch (default: True)')
    parser.add_argument('--track', action='store_true', help='Track with Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default=None, help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity')
    parser.add_argument('--capture-video', action='store_true', help='Capture videos')
    
    args = parser.parse_args()
    
    kwargs = {
        'seed': args.seed,
    }
    
    if args.total_timesteps is not None:
        kwargs['total_timesteps'] = args.total_timesteps
    if args.learning_rate is not None:
        kwargs['learning_rate'] = args.learning_rate
    if args.num_envs is not None:
        kwargs['num_envs'] = args.num_envs
    if args.num_steps is not None:
        kwargs['num_steps'] = args.num_steps
    if args.gamma is not None:
        kwargs['gamma'] = args.gamma
    if args.gae_lambda is not None:
        kwargs['gae_lambda'] = args.gae_lambda
    if args.update_epochs is not None:
        kwargs['update_epochs'] = args.update_epochs
    if args.num_minibatches is not None:
        kwargs['num_minibatches'] = args.num_minibatches
    if args.norm_adv is not None:
        kwargs['norm_adv'] = args.norm_adv
    if args.anneal_lr is not None:
        kwargs['anneal_lr'] = args.anneal_lr
    if args.clip_vloss is not None:
        kwargs['clip_vloss'] = args.clip_vloss
    if args.ent_coef is not None:
        kwargs['ent_coef'] = args.ent_coef
    if args.vf_coef is not None:
        kwargs['vf_coef'] = args.vf_coef
    if args.max_grad_norm is not None:
        kwargs['max_grad_norm'] = args.max_grad_norm
    if args.target_kl is not None:
        kwargs['target_kl'] = args.target_kl
    if args.cuda is not None:
        kwargs['cuda'] = args.cuda
    if args.torch_deterministic is not None:
        kwargs['torch_deterministic'] = args.torch_deterministic
    if args.track:
        kwargs['track'] = True
    if args.wandb_project is not None:
        kwargs['wandb_project_name'] = args.wandb_project
    if args.wandb_entity is not None:
        kwargs['wandb_entity'] = args.wandb_entity
    if args.capture_video:
        kwargs['capture_video'] = True
    
    if args.algorithm == 'ppo':
        if args.clip_coef is not None:
            kwargs['clip_coef'] = args.clip_coef
    elif args.algorithm in ['ppo_br', 'simple_ppo_br']:
        # Use paper notation (epsilon_0, lambda_1, lambda_2)
        if args.epsilon_0 is not None:
            kwargs['epsilon_0'] = args.epsilon_0
        elif args.base_clip_coef is not None:
            kwargs['epsilon_0'] = args.base_clip_coef  # Backward compatibility
        if args.algorithm == 'ppo_br' and args.value_clip_coef is not None:
            kwargs['value_clip_coef'] = args.value_clip_coef
        if args.lambda_1 is not None:
            kwargs['lambda_1'] = args.lambda_1
        elif args.lambda1 is not None:
            kwargs['lambda_1'] = args.lambda1  # Backward compatibility
        if args.lambda_2 is not None:
            kwargs['lambda_2'] = args.lambda_2
        elif args.lambda2 is not None:
            kwargs['lambda_2'] = args.lambda2  # Backward compatibility
        if args.reward_window_size is not None:
            kwargs['reward_window_size'] = args.reward_window_size
        if args.algorithm == 'simple_ppo_br':
            if args.psi_method is not None:
                kwargs['psi_method'] = args.psi_method
            if args.delta_r_method is not None:
                kwargs['delta_r_method'] = args.delta_r_method
    
    if args.num_seeds > 1:
        print(f"\n{'='*60}")
        print(f"Running {args.algorithm.upper()} with {args.num_seeds} seeds")
        print(f"Seeds: {args.start_seed} to {args.start_seed + args.num_seeds - 1}")
        print(f"{'='*60}\n")
        
        results = {}
        for seed_offset in range(args.num_seeds):
            current_seed = args.start_seed + seed_offset
            print(f"\n{'='*60}")
            print(f"Seed {seed_offset + 1}/{args.num_seeds}: seed={current_seed}")
            print(f"{'='*60}")
            
            kwargs['seed'] = current_seed
            
            if args.env.lower() == 'all':
                seed_results = run_all_experiments(algorithm=args.algorithm, **kwargs)
                results[current_seed] = seed_results
            else:
                try:
                    agent = run_experiment(env_id=args.env, algorithm=args.algorithm, **kwargs)
                    results[current_seed] = {'success': True, 'agent': agent}
                except Exception as e:
                    print(f"\n✗ Failed with seed {current_seed}: {e}")
                    results[current_seed] = {'success': False, 'error': str(e)}
        
        print(f"\n{'='*60}")
        print("SUMMARY (All Seeds)")
        print(f"{'='*60}")
        for seed, result in results.items():
            if args.env.lower() == 'all':
                print(f"\nSeed {seed}:")
                for env_id, env_result in result.items():
                    status = "✓" if env_result.get('success') else "✗"
                    print(f"  {status} {env_id}")
            else:
                status = "✓" if result.get('success') else "✗"
                print(f"{status} Seed {seed}: {args.env}")
        
        return results
    else:
        if args.env.lower() == 'all':
            run_all_experiments(algorithm=args.algorithm, **kwargs)
        else:
            run_experiment(env_id=args.env, algorithm=args.algorithm, **kwargs)


if __name__ == '__main__':
    main()

