# RL Framework

CleanRL 기반의 모듈화된 PPO와 PPO-BR 알고리즘 구현

## 구조

```
RL_Framework/
├── algorithms/          # 알고리즘 구현
│   ├── ppo.py          # PPO 클래스
│   └── ppo_br.py       # PPO-BR 클래스
├── utils/              # 유틸리티 함수
│   ├── env_utils.py    # 환경 생성 함수
│   └── network_utils.py # 네트워크 정의
├── main.py             # 통합 실행 인터페이스
└── README.md
```

## 사용법

```bash
# PPO-BR on CartPole
python main.py --algorithm ppo_br --env CartPole-v0 --seed 42

# PPO on HalfCheetah
python main.py --algorithm ppo --env HalfCheetah-v4 --seed 42

# 모든 환경에서 실행
python main.py --algorithm ppo_br --env all
```

## 지원 환경

- **Discrete**: CartPole-v0, LunarLander-v3
- **Continuous**: Hopper-v4, HalfCheetah-v4, Walker2d-v4, Humanoid-v4

## 환경별 설정

| 환경 | Timesteps | Envs | Steps | Ent Coef | Lambda1 | Lambda2 | Window |
|------|-----------|------|-------|----------|---------|---------|--------|
| CartPole | 500K | 4 | 128 | 0.01 | 0.5 | 0.3 | 10 |
| LunarLander | 1M | 4 | 128 | 0.01 | 0.7 | 0.3 | 10 |
| Hopper | 3M | 1 | 2048 | 0.0 | 0.3 | 0.2 | 20 |
| HalfCheetah | 3M | 1 | 2048 | 0.0 | 0.5 | 0.3 | 10 |
| Walker2d | 3M | 1 | 2048 | 0.0 | 0.5 | 0.3 | 10 |
| Humanoid | 10M | 1 | 2048 | 0.0 | 0.5 | 0.3 | 10 |

## 하이퍼파라미터

### 공통 (PPO & PPO-BR)
- `learning_rate`: 3e-4
- `gamma`: 0.99
- `gae_lambda`: 0.95
- `base_clip_coef`: 0.2

### PPO-BR 전용
- `lambda1`: 0.5 (LunarLander: 0.7, Hopper: 0.3)
- `lambda2`: 0.3 (Hopper: 0.2)
- `reward_window_size`: 10 (Hopper: 20)
- `min_clip_coef`: 0.1 (Hopper: 0.15)
- `max_clip_coef`: 0.4 (Hopper: 0.3)

## 참고

- CleanRL: [High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms](https://arxiv.org/pdf/2111.08819)
- PPO-BR: [Dual-Signal Entropy-Reward Adaptation for Trust Region Policy Optimization](https://arxiv.org/pdf/2505.17714)
