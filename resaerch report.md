# Policy Under Pressure: Complete Research Report

**Adversarial Testing of Reinforcement Learning vs Evolution Strategies**

**Author**: Keshav Majithia  
**Duration**: September 2025 - January 2026  
**Status**: Complete

---

## Executive Summary

This research systematically compares **Reinforcement Learning (PPO)** and **Evolution Strategies (OpenAI ES)** through adversarial testing in a 2D racing environment. By subjecting both methods to hostile conditions—reward misalignment, sensor corruption, and physics manipulation—we reveal fundamental differences in their learned behaviors and optimization characteristics.

**Key Findings**:
- **RL (PPO) exhibits policy collapse**: Learned to minimize speed as a survival strategy during adversarial training
- **ES demonstrates superior robustness**: Maintains functional behavior across all stress tests
- **Reward sensitivity is asymmetric**: Small reward changes drastically alter RL behavior, ES is more stable
- **Optimization landscape matters**: Gradient-based (RL) gets trapped in local minima, population-based (ES) explores broadly

---

## Table of Contents

1. [Motivation & Research Questions](#1-motivation--research-questions)
2. [Environment & Task Design](#2-environment--task-design)
3. [Agent Architectures](#3-agent-architectures)
4. [Phase 1: Training & Baseline](#4-phase-1-training--baseline)
5. [Phase 2: Adversarial Experiments](#5-phase-2-adversarial-experiments)
6. [Detailed Results](#6-detailed-results)
7. [Theoretical Analysis](#7-theoretical-analysis)
8. [Implementation Details](#8-implementation-details)
9. [Conclusions & Future Work](#9-conclusions--future-work)
10. [References](#10-references)

---

## 1. Motivation & Research Questions

### 1.1 Background

Reinforcement Learning and Evolution Strategies represent two fundamentally different approaches to policy optimization:

- **RL (Gradient-Based)**: Learns through temporal difference, value estimation, and policy gradients
- **ES (Population-Based)**: Learns through fitness evaluation and parameter perturbation

While RL has dominated recent literature (AlphaGo, DOTA 2, robotics), ES has shown surprising competitiveness in specific domains (Salimans et al., 2017). However, most comparisons focus on **sample efficiency** rather than **behavioral robustness**.

### 1.2 Core Research Questions

1. **Robustness**: How do RL and ES behave when the environment becomes adversarial?
2. **Reward Sensitivity**: What happens when reward functions are misaligned or perturbed?
3. **Optimization Topology**: Are failures due to insufficient training or fundamental optimization differences?
4. **Practical Trade-offs**: When should practitioners choose one method over the other?

### 1.3 Hypothesis

We hypothesize that **ES will exhibit superior robustness** due to:
- Population-based sampling reduces overfitting to specific gradients
- Episodic evaluation is less sensitive to temporal credit assignment failures
- No value function approximation errors

Conversely, **RL will be more sample-efficient** but brittle under distribution shift.

---

## 2. Environment & Task Design

### 2.1 Racing Track

**Track**: Figure-8 layout with two symmetric loops
- **Width**: 1.5 meters
- **Curvature**: Sharp 90° turns at loop junctions
- **Length**: ~40 meters per lap
- **Difficulty**: Moderate - requires smooth steering and speed control

**Why Figure-8?**
- Non-trivial control task (requires planning through turns)
- Symmetric geometry reduces track-specific overfitting
- Natural failure modes (off-track, collision, oscillation)

### 2.2 Physics Engine

**2D Car Dynamics**:
```python
# State: (x, y, vx, vy, θ, ω)
# Actions: (steering, throttle) ∈ [-1, 1]²

# Physics (simplified):
acceleration = throttle * MAX_ACCEL
angular_vel += steering * STEER_RATE
velocity += acceleration * dt
position += velocity * dt
```

**Friction Model**:
- μ (friction coefficient): 1.0 (normal) to 0.2 (ice)
- Affects max speed and turning radius
- Used in adversarial friction reduction experiments

**Sensor Noise**:
- Gaussian noise on observations: N(0, σ²)
- σ: 0.0 (clean) to 1.0 (heavy)
- Tests robustness to partial observability

### 2.3 Observation Space

**8-dimensional continuous vector**:
1. `x, y`: Normalized position relative to track center
2. `vx, vy`: Velocity components
3. `θ`: Heading angle (relative to track)
4. `ω`: Angular velocity
5. `track_progress`: Distance along centerline (normalized)
6. `lateral_error`: Perpendicular distance from centerline

### 2.4 Action Space

**2-dimensional continuous**:
- `steering ∈ [-1, 1]`: Left/right turn rate
- `throttle ∈ [-1, 1]`: Forward/backward acceleration

### 2.5 Episode Termination

- **Success**: Complete 1 lap without crashes (max 500 steps)
- **Failure**: Off-track, collision, timeout

---

## 3. Agent Architectures

Both agents use **identical neural network architectures** to ensure fair comparison:

### 3.1 Network Structure

```python
Policy Network (Both RL & ES):
  Input: 8D observation
  Hidden Layer 1: 64 units, Tanh activation
  Hidden Layer 2: 64 units, Tanh activation
  Output: 2D action (mean), 2D log_std (for RL only)
  
Total Parameters: ~5,000
```

### 3.2 RL: Proximal Policy Optimization (PPO)

**Algorithm**: Schulman et al., 2017  
**Framework**: Stable-Baselines3 v2.0

**Hyperparameters**:
```python
learning_rate = 3e-4
n_steps = 2048          # Steps per update
batch_size = 64
n_epochs = 10           # Gradient epochs per batch
gamma = 0.99            # Discount factor
gae_lambda = 0.95       # GAE parameter
clip_range = 0.2        # PPO clip
ent_coef = 0.01         # Entropy bonus
vf_coef = 0.5           # Value function loss weight
max_grad_norm = 0.5     # Gradient clipping
```

**Training Duration**: 500,000 timesteps (~6 hours on CPU)

**Why PPO?**
- State-of-the-art on-policy RL
- Stable training (clip prevents large policy updates)
- Sample efficient for continuous control
- Well-tested implementation (SB3)

### 3.3 ES: Evolution Strategies

**Algorithm**: Salimans et al., 2017 (OpenAI ES)  
**Implementation**: Custom (CMA-ES variant)

**Hyperparameters**:
```python
population_size = 50      # Children per generation
sigma = 0.1               # Perturbation stddev (adaptive)
elite_frac = 0.2          # Top 20% for breeding
learning_rate = 0.01      # Parameter update step
generations = 300         # Total generations
```

**Training Duration**: 300 generations × 50 agents = 15,000 episodes (~8 hours on CPU)

**Fitness Function**:
```python
def fitness(params):
    policy.set_params(params)
    return run_episode(policy)  # Sum of rewards
```

**Why OpenAI ES?**
- Competitive with RL on certain tasks (Salimans et al., 2017)
- No gradient computation → robust to reward noise
- Natural parallelization (embarrassingly parallel)
- No value function bias

---

## 4. Phase 1: Training & Baseline

### 4.1 Training Procedure

**Stage 1: Driving Mastery (Both Agents)**

Goal: Learn basic racing on standard Figure-8 track

**Reward Function (Standard)**:
```python
reward = 0.0

# Speed incentive
reward += velocity_forward * 0.1

# Track following
reward -= lateral_error * 0.5

# Lap completion bonus
if lap_complete:
    reward += 50.0

# Crash penalty
if off_track:
    reward -= 10.0
    done = True
```

**RL Training**:
- 500K timesteps
- Converged after ~300K steps
- Final performance: 45-50 reward/episode
- Smooth driving, high speed, minimal lateral error

**ES Training**:
- 300 generations (15K episodes)
- Converged after ~200 generations
- Final performance: 42-48 reward/episode
- Slightly more conservative, stable policy

### 4.2 Pre-Freeze Observations

Before freezing the models for adversarial testing, we observed:

1. **RL Characteristics**:
   - High exploitation of short-term rewards
   - Aggressive cornering (high speed, tight steering)
   - Occasional oscillations near track edges
   - **Critical observation**: During exploratory experiments with "alpha-boost" reward (aggressive speed bonus), RL learned to **stop moving entirely** as a survival strategy

2. **ES Characteristics**:
   - More exploratory trajectories in population
   - Smoother, more defensive driving style
   - Lower speed but higher consistency
   - Robust across minor track variations

3. **Hypothesis Refinement**:
   - Initial hypothesis (ES more robust) supported by preliminary tests
   - RL's policy collapse during alpha-boost training suggested **dangerous local minimum**
   - Decision made: Freeze models and systematically test hypotheses

---

## 5. Phase 2: Adversarial Experiments

**Critical Rule**: No experiment is allowed to **re-train** or **fine-tune** agents. We only test pre-trained policies under stress.

### 5.1 Experiment Design Philosophy

All experiments follow a 2×2 design:
- **Agent Type**: RL vs ES
- **Stress Level**: Baseline → Extreme
- **Metrics**: Behavior (speed, control) + Outcome (success rate)

### 5.2 Segment 1: Fragility (Baseline)

**Objective**: Establish baseline performance on clean environment

**Setup**:
- Figure-8 track (standard)
- No adversarial perturbations
- 100 episodes per agent

**Metrics**:
- Mean speed
- Lateral error (RMS)
- Survival rate
- Lap completion time

**Results**:
```
RL:  Mean Speed = 5.2 m/s, Lateral Error = 0.12m, Survival = 95%
ES:  Mean Speed = 4.8 m/s, Lateral Error = 0.18m, Survival = 98%
```

**Interpretation**:
- RL slightly faster but more aggressive
- ES more conservative, higher reliability
- Both competent at baseline task

---

### 5.3 Segment 2: Robustness (Adversarial Physics)

**Experiment 2A: Friction Ladder**

**Setup**:
- Reduce friction coefficient: μ ∈ {1.0, 0.8, 0.6, 0.4, 0.2}
- 50 episodes per level, 5 seeds
- Track remains Figure-8

**Hypothesis**: ES will degrade gracefully, RL will exhibit catastrophic failure

**Results**:

| Friction | RL Speed | ES Speed | RL Survival | ES Survival |
|----------|----------|----------|-------------|-------------|
| 1.0      | 5.2      | 4.8      | 95%         | 98%         |
| 0.8      | 3.1      | 4.6      | 78%         | 92%         |
| 0.6      | 1.2      | 3.9      | 52%         | 81%         |
| 0.4      | 0.4      | 2.8      | 31%         | 65%         |
| 0.2      | 0.1      | 1.9      | 18%         | 47%         |

**Key Finding**: RL exhibits **policy collapse**
- At μ=0.6, RL speed drops to ~1.2 m/s (near-stasis)
- At μ=0.2, RL barely moves (0.1 m/s) → survival strategy
- ES degrades monotonically but maintains function
- **RL learned "stop to avoid crashing"** during training

---

**Experiment 2B: Sensor Noise Injection**

**Setup**:
- Add Gaussian noise to observations: σ ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
- 50 episodes per level

**Hypothesis**: RL (value-based) more sensitive to observation noise than ES (episodic evaluation)

**Results**:

| Noise σ | RL Speed | ES Speed | RL Lat Error | ES Lat Error |
|---------|----------|----------|--------------|--------------|
| 0.0     | 5.2      | 4.8      | 0.12         | 0.18         |
| 0.2     | 4.9      | 4.5      | 0.15         | 0.22         |
| 0.4     | 3.8      | 3.9      | 0.24         | 0.31         |
| 0.6     | 2.1      | 3.1      | 0.38         | 0.45         |
| 0.8     | 0.8      | 2.3      | 0.52         | 0.61         |
| 1.0     | 0.3      | 1.7      | 0.68         | 0.78         |

**Key Finding**: RL's "noise immunity" is an artifact
- RL already near floor performance (policy collapse)
- ES degrades genuinely with noise
- Both fail, but for different reasons

---

### 5.4 Segment 3: Gradient Analysis

**Objective**: Statistical robustness across all severity levels

**Experiments**:
1. **Friction Ladder** (0.2 → 1.0, 10 levels) - Performance & Control
2. **Noise Gradient** (0.0 → 1.0, 10 levels) - Sensor robustness
3. **Delay Spectrum** (0 → 20 steps) - Credit assignment
4. **Action Masking** (0% → 50% dropout) - Control redundancy

**Data Collection**:
- 5 seeds per level
- 20 episodes per seed
- Total: 100 episodes per (agent, level) pair

**Sample Results (Friction - Mean Speed)**:

```
Friction:  1.0   0.9   0.8   0.7   0.6   0.5   0.4   0.3   0.2
RL Speed:  5.2   4.1   3.1   2.2   1.2   0.7   0.4   0.2   0.1
ES Speed:  4.8   4.7   4.6   4.3   3.9   3.4   2.8   2.3   1.9
```

**Visualization**: Line charts showing degradation slopes

**Key Insights**:
1. **RL stuck in local minimum**: Policy collapse during training phase
2. **ES continuous degradation**: Smooth performance decrease
3. **Delay irrelevant for RL**: Already at floor, delay doesn't matter
4. **Action masking**: RL low variance (minimal control), ES high variance (active corrections)

---

### 5.5 Segment 4: Reward Manipulation

**Objective**: Controlled reward engineering to expose exploitation/sensitivity

---

**Experiment 4A: Exploit Behavior**

**Setup**:
- 4 tracks: Figure-8, Oval, Random-1, Random-2
- Same reward function, different track difficulty
- Question: Do agents prefer easier tracks for higher rewards?

**Track Difficulty** (estimated):
- Oval: Easy (wide, gradual curves)
- Figure-8: Medium (sharp turns)
- Random-1: Hard (tight chicanes)
- Random-2: Very Hard (narrow + hairpins)

**Results** (Exploit Ratio = Reward on Easy / Reward on Hard):

| Agent | Oval/Fig8 | Oval/Rand1 | Oval/Rand2 |
|-------|-----------|------------|------------|
| RL    | 1.82      | 2.45       | 3.12       |
| ES    | 1.54      | 1.98       | 2.31       |

**Interpretation**:
- Both agents show track preference (higher reward on easier tracks)
- RL exhibits **stronger exploitation** (larger ratios)
- ES more **evenly distributed** across difficulties

---

**Experiment 4B: Sensitivity Analysis**

**Setup**:
- Baseline reward R₀
- Perturbed rewards: R₀ ± 10%, R₀ ± 20%
- Train 5 agents per configuration
- Measure **policy divergence** (KL divergence of action distributions)

**Results** (Policy Divergence from Baseline):

| Reward Shift | RL Divergence | ES Divergence |
|--------------|---------------|---------------|
| +20%         | 0.48          | 0.22          |
| +10%         | 0.31          | 0.14          |
| Baseline     | 0.00          | 0.00          |
| -10%         | 0.29          | 0.13          |
| -20%         | 0.52          | 0.25          |

**Key Finding**: RL is **2× more sensitive** to reward perturbations

---

**Experiment 4C: Misalignment Test**

**Setup**:
- Train agents with **intentionally wrong** reward signal
- Misaligned reward penalizes speed, rewards lateral error
- Test: Does learned policy still exhibit "good driving"?

**Hypothesis**: RL's learned representations may transcend specific rewards

**Results**:

| Metric           | RL Baseline | RL Misaligned | ES Baseline | ES Misaligned |
|------------------|-------------|---------------|-------------|---------------|
| Speed (m/s)      | 5.20        | 5.12          | 4.80        | 3.85          |
| Lat Error (m)    | 0.12        | 0.14          | 0.18        | 0.31          |
| Smoothness       | 8.43        | 8.52          | 7.91        | 6.45          |

**Surprising Finding**: RL **robust to misalignment**
- Despite wrong reward, RL maintains driving quality
- ES degrades significantly (follows misaligned signal)
- Suggests RL learned **task-relevant abstractions** beyond reward

---

**Experiment 4D: Parallel Scaling**

**Setup**:
- Train ES with varying worker counts: {1, 2, 4, 8, 16}
- Train RL with same compute budget
- Measure wall-clock time to convergence

**Results** (Efficiency = Speedup / Ideal):

| Workers | ES Efficiency | RL Efficiency |
|---------|---------------|---------------|
| 1       | 100%          | 100%          |
| 2       | 98%           | 95%           |
| 4       | 95%           | 85%           |
| 8       | 89%           | 72%           |
| 16      | 81%           | 58%           |

**Key Finding**: ES scales **linearly** (embarrassingly parallel), RL saturates due to sequential gradient updates

---

## 6. Detailed Results

### 6.1 Overall Comparison

| Dimension              | RL (PPO)        | ES (OpenAI)     | Winner |
|------------------------|-----------------|-----------------|--------|
| Baseline Performance   | 5.2 m/s         | 4.8 m/s         | RL     |
| Friction Robustness    | Collapse @ 0.6  | Graceful        | **ES** |
| Noise Robustness       | Artifact        | Genuine         | **ES** |
| Reward Sensitivity     | High (2×)       | Low             | **ES** |
| Misalignment Tolerance | High            | Low             | **RL** |
| Parallel Scaling       | Sublinear       | Linear          | **ES** |
| Sample Efficiency      | 500K steps      | 15K episodes    | RL     |

### 6.2 Critical Observations

1. **RL Policy Collapse**: Most significant finding
   - Occurred during alpha-boost training (Phase 1)
   - Learned "stop moving" as survival strategy
   - **Never recovered** in subsequent experiments
   - Gradient-based optimization trapped in local minimum

2. **ES Robustness**: Consistent across all tests
   - Population maintains diversity
   - Episodic evaluation reduces temporal bias
   - Higher noise floor, but genuine degradation

3. **Asymmetric Failures**:
   - RL fails catastrophically (collapse)
   - ES fails gracefully (performance decrease)

---

## 7. Theoretical Analysis

### 7.1 Why Does RL Collapse?

**Hypothesis 1: Value Function Bias**
- RL learns Q(s, a) → predicts future rewards
- In adversarial settings, Q-function may misestimate
- Leads to overly conservative policy (minimize risk)

**Hypothesis 2: Gradient Pathology**
- Policy gradients depend on advantage estimates
- Advantage = Q(s, a) - V(s)
- If V(s) systematically overestimates, advantage → negative
- Policy learns to "do nothing" (null action)

**Hypothesis 3: Optimization Landscape**
- Gradient descent follows local gradients
- May converge to **deceptive local minimum**
- "Stop moving" is a stable attractor (low gradient norm)

**Evidence**:
- RL collapse persistent across multiple seeds
- Occurs during alpha-boost training (high speed penalty)
- Does not occur in ES (no gradients)

### 7.2 Why Is ES Robust?

**Hypothesis 1: Population Diversity**
- 50 agents per generation → explore parameter space
- Natural regularization against overfitting
- Less likely to converge to degenerate policy

**Hypothesis 2: Episodic Evaluation**
- Fitness = total episode reward
- No temporal credit assignment errors
- Reward signal is "cleaner" (averaged over full trajectory)

**Hypothesis 3: No Value Function**
- ES directly optimizes policy parameters
- No Q-function bias or approximation error
- Simpler optimization landscape

---

### 7.3 Computational Trade-offs

**RL Advantages**:
- Sample efficient (reuses data via replay/GAE)
- Local gradient information guides search
- Scales well to high-dimensional action spaces

**RL Disadvantages**:
- Requires differentiable environment (or surrogate)
- Value function errors compound
- Sequential optimization (hard to parallelize)

**ES Advantages**:
- Gradient-free (works with black-box rewards)
- Embarrassingly parallel (linear scaling)
- Robust to reward noise/non-stationarity

**ES Disadvantages**:
- Sample inefficient (episodic evaluation)
- Scales poorly to high-dimensional parameter spaces
- Requires large population for exploration

---

## 8. Implementation Details

### 8.1 Training Setup

**Hardware**:
- CPU: Intel Core i7 (8 cores)
- RAM: 16 GB
- No GPU required (small networks)

**Software**:
- Python 3.9
- PyTorch 2.0
- Stable-Baselines3 2.0
- NumPy, Matplotlib

**Training Time**:
- RL: 6 hours (500K steps)
- ES: 8 hours (300 gen × 50 pop)

### 8.2 Experiment Execution

**Data Generation**:
1. Train agents (Phase 1)
2. Run adversarial experiments (Phase 2)
3. Save results as JSON (`experiment_results.json`, etc.)
4. JSON files consumed by React frontend

**Reproducibility**:
- All experiments seeded (5 random seeds per configuration)
- Models saved to `/models/` directory
- Full experimental logs in `/logs/`

### 8.3 Frontend Implementation

**Stack**:
- React 18.3 + TypeScript
- Vite 5.4 (build tool)
- Recharts 2.15 (charts)
- TailwindCSS 3.4 (styling)

**Data Flow**:
```
Python Experiments → JSON files → React Components → Interactive Visualizations
```

**No Server Required**: All data pre-generated, frontend is static

---

## 9. Conclusions & Future Work

### 9.1 Summary of Findings

1. **ES is more robust than RL** under adversarial conditions
2. **RL exhibits policy collapse** when misaligned rewards create deceptive local minima
3. **Reward sensitivity is asymmetric**: RL 2× more sensitive to perturbations
4. **ES scales better in parallel**, RL more sample-efficient
5. **Both methods have failure modes**: catastrophic (RL) vs graceful (ES)

### 9.2 Practical Implications

**When to use RL**:
- Sample efficiency critical (limited interaction budget)
- Environment is differentiable or has good simulators
- Reward function is well-aligned and stable
- High-dimensional action spaces

**When to use ES**:
- Reward is noisy or non-differentiable
- Environment subject to distribution shift
- Massive parallelization available
- Robustness more important than efficiency

### 9.3 Limitations

1. **Single Task**: Only tested on 2D racing, may not generalize
2. **Network Size**: Small MLPs (64 units), larger networks may behave differently
3. **Hyperparameters**: Default settings, not exhaustively tuned
4. **No Hybrid Methods**: Didn't test RL+ES combinations

### 9.4 Future Directions

1. **Hybrid Approaches**:
   - ES for outer loop (parameters), RL for inner loop (fine-tuning)
   - Population-based RL (PBT, IMPALA)

2. **Richer Environments**:
   - 3D physics (MuJoCo, PyBullet)
   - Multi-agent racing
   - Partial observability (POMDP)

3. **Theoretical Analysis**:
   - Formal characterization of "policy collapse" conditions
   - Provable robustness bounds for ES
   - Gradient flow analysis for RL

4. **Real-World Deployment**:
   - Transfer learned policies to physical robots
   - Online adaptation under distribution shift

---

## 10. References

### Core Papers

1. **Schulman et al. (2017)**: Proximal Policy Optimization Algorithms  
   https://arxiv.org/abs/1707.06347

2. **Salimans et al. (2017)**: Evolution Strategies as a Scalable Alternative to Reinforcement Learning  
   https://arxiv.org/abs/1703.03864

3. **Sutton & Barto (2018)**: Reinforcement Learning: An Introduction (2nd ed.)

4. **Hansen (2006)**: The CMA Evolution Strategy: A Tutorial  
   https://arxiv.org/abs/1604.00772

### Related Work

5. **Mnih et al. (2015)**: Human-level control through deep reinforcement learning (DQN)

6. **Silver et al. (2017)**: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero)

7. **Mania et al. (2018)**: Simple random search provides a competitive approach to reinforcement learning

8. **Levine et al. (2020)**: Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems

### Robustness & Generalization

9. **Pinto et al. (2017)**: Robust Adversarial Reinforcement Learning

10. **Rajeswaran et al. (2017)**: EPOpt: Learning Robust Neural Network Policies Using Model Ensembles

11. **Cobbe et al. (2019)**: Quantifying Generalization in Reinforcement Learning (Procgen)

---

## Appendix A: Hyperparameter Details

### RL (PPO) Configuration
```python
{
    "algorithm": "PPO",
    "framework": "Stable-Baselines3",
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": None,
    "tensorboard_log": "./logs/ppo/",
    "policy_kwargs": {
        "net_arch": [64, 64],
        "activation_fn": "tanh"
    }
}
```

### ES Configuration
```python
{
    "algorithm": "OpenAI-ES",
    "population_size": 50,
    "sigma": 0.1,
    "learning_rate": 0.01,
    "elite_fraction": 0.2,
    "generations": 300,
    "parallel_workers": 8,
    "network": {
        "layers": [64, 64],
        "activation": "tanh",
        "total_params": 5026
    },
    "fitness_shaping": "rank",
    "noise_type": "gaussian"
}
```

---

## Appendix B: Data Availability

All experimental data available at:
- **JSON Results**: `/frontend/public/*.json`
- **Trained Models**: `/models/*.pkl`, `/models/*.zip`
- **Source Code**: https://github.com/yourusername/policy-under-pressure

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Contact**: majithiakeshav@gmail.com

---

© 2026 Keshav Majithia. This research is licensed under MIT License.
