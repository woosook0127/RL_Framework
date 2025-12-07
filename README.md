# RL Framework

CleanRL 기반의 모듈화된 PPO와 PPO-BR 알고리즘 구현

## 설치 및 환경 설정

### 1. Python 환경 생성 (권장)

```bash
# Conda를 사용하는 경우
conda create -n rl_ppo python=3.10
conda activate rl_ppo

# 또는 venv를 사용하는 경우
python -m venv rl_ppo
source rl_ppo/bin/activate  # Linux/Mac
# 또는
rl_ppo\Scripts\activate  # Windows
```

### 2. 의존성 설치

```bash
# 프로젝트 디렉토리로 이동
cd RL_Framework

# requirements.txt를 사용하여 설치
pip install -r requirements.txt
```

### 빠른 시작

환경 설정이 완료되면 다음 명령으로 테스트할 수 있습니다:

```bash
# 간단한 테스트 (CartPole, 빠른 수렴)
python main.py --algorithm ppo_br --env CartPole-v0

# MuJoCo 환경 테스트 (Hopper)
python main.py --algorithm ppo_br --env Hopper-v5
```

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
python main.py --algorithm ppo_br --env CartPole-v0

# PPO on HalfCheetah
python main.py --algorithm ppo --env HalfCheetah-v5

# Simple PPO-BR on LunarLander
python main.py --algorithm simple_ppo_br --env LunarLander-v3

# 모든 환경 실행
python main.py --algorithm ppo_br --env all

# 여러 시드로 실험
python main.py --algorithm ppo_br --env Hopper-v5 --num-seeds 5 --start-seed 1
```

## 실험 환경

- **Discrete**: CartPole-v0, LunarLander-v3
- **Continuous**: Hopper-v4, HalfCheetah-v5, Walker2d-v5, Humanoid-v5

## 환경별 설정

| 환경 | Timesteps | Envs | Steps | Ent Coef | Lambda1 | Lambda2 | Window |
|------|-----------|------|-------|----------|---------|---------|--------|
| CartPole | 500K | 4 | 128 | 0.01 | 0.5 | 0.3 | 10 |
| LunarLander | 1M | 4 | 128 | 0.02 | 0.8 | 0.3 | 15 |
| Hopper | 3M | 1 | 2048 | 5e-4 | 0.65 | 0.2 | 22 |
| HalfCheetah | 3M | 1 | 2048 | 5e-4 | 0.7 | 0.2 | 20 |
| Walker2d | 3M | 1 | 2048 | 0.0 | 0.65 | 0.28 | 25 |
| Humanoid | 10M | 1 | 2048 | 0.0 | 0.6 | 0.3 | 40 |

## 하이퍼파라미터

### 공통 (PPO & PPO-BR)
- `learning_rate`: 3e-4
- `gamma`: 0.99
- `gae_lambda`: 0.95
- `base_clip_coef`: 0.2

### PPO-BR 전용
- `lambda1`: 0.5
- `lambda2`: 0.3
- `reward_window_size`: 10
- `min_clip_coef`: 0.1
- `max_clip_coef`: 0.4

## 문제 해결

### MuJoCo 설치 문제

MuJoCo 설치에 문제가 있는 경우:

```bash
# MuJoCo 재설치
pip install --upgrade mujoco gymnasium[mujoco]

# 또는 시스템 패키지로 설치 (Linux)
# sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3
```

## 참고

- CleanRL: [High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms](https://arxiv.org/pdf/2111.08819)
- PPO-BR: [Dual-Signal Entropy-Reward Adaptation for Trust Region Policy Optimization](https://arxiv.org/pdf/2505.17714)
