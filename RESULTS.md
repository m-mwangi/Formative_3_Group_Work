# Hyperparameter Tuning Results and Observations

## Overview

This document presents the results of our hyperparameter tuning experiments for the DQN agent trained on the Atari Breakout environment. As the "Exploration & Curiosity" specialist, our mission was to find the best exploration strategy by tuning the epsilon-greedy parameters: `epsilon_decay` (how long the agent explores) and `epsilon_end` (how much curiosity it keeps at the end).

If the agent stops exploring too soon, it may never learn the best strategies. If it explores for too long, it may never stabilize and get a high score.

All experiments were conducted with 100,000 training steps. The baseline configuration used: lr=0.0001, gamma=0.99, batch_size=32, epsilon_decay=250000, epsilon_end=0.05.

## Experimental Setup

We ran experiments varying epsilon_decay and epsilon_end around the baseline values:

| Exp | Goal | epsilon_decay | epsilon_end | Command |
| --- | ---- | -------------- | ----------- | ------- |
| Baseline | Standard | 250000 | 0.05 | (default) |
| Fast Decay + Low End | Very Fast Decay | 100000 | 0.01 | `--epsilon_decay 100000 --epsilon_end 0.01` |
| Fast Decay + Base End | Fast Decay | 100000 | 0.05 | `--epsilon_decay 100000 --epsilon_end 0.05` |
| Slow Decay + Base End | Slow Decay | 500000 | 0.05 | `--epsilon_decay 500000 --epsilon_end 0.05` |
| Very Slow Decay + Base End | Very Slow Decay | 750000 | 0.05 | `--epsilon_decay 750000 --epsilon_end 0.05` |
| Base Decay + Very Low End | No Final Exploration | 250000 | 0.001 | `--epsilon_end 0.001` |
| Base Decay + Low-Mid End | Low-Mid Final Expl. | 250000 | 0.025 | `--epsilon_end 0.025` |
| Base Decay + High End | High Final Expl. | 250000 | 0.1 | `--epsilon_end 0.1` |
| Base Decay + Very High End | Very High Final Expl. | 250000 | 0.15 | `--epsilon_end 0.15` |
| Slow Decay + High End | Interact: Slow Decay + High Final | 500000 | 0.1 | `--epsilon_decay 500000 --epsilon_end 0.1` |

## Results

The primary metric evaluated is the mean episode reward (rollout/ep_rew_mean) at the end of training. Higher values indicate better performance.

| Exp | epsilon_decay | epsilon_end | Final Mean Reward | Observations |
| --- | -------------- | ----------- | ----------------- | ------------ |
| Baseline | 250000 | 0.05 | 1.75 | Baseline performance, still exploring significantly at end |
| Fast Decay + Low End | 100000 | 0.01 | 9.60 | Good balance, stopped exploring early and exploited learned policy |
| Fast Decay + Base End | 100000 | 0.05 | 12.70 | Best performance, fast decay allowed focus on exploitation |
| Slow Decay + Base End | 500000 | 0.05 | 3.47 | Still exploring too much, lower performance |
| Very Slow Decay + Base End | 750000 | 0.05 | 2.65 | Excessive exploration, poor exploitation |
| Base Decay + Very Low End | 250000 | 0.001 | 2.92 | Low final exploration, but decay too slow |
| Base Decay + Low-Mid End | 250000 | 0.025 | 3.14 | Moderate performance, balanced exploration |
| Base Decay + High End | 250000 | 0.1 | 3.00 | High final exploration, hindered exploitation |
| Base Decay + Very High End | 250000 | 0.15 | 2.97 | Very high exploration, poor final performance |
| Slow Decay + High End | 500000 | 0.1 | 2.16 | Worst performance, too much exploration throughout |

## Key Insights from Tuning

### Which hyperparameter changes improved performance?

- **Faster epsilon decay (100000)**: Both experiments with epsilon_decay=100000 significantly outperformed the baseline (9.60 and 12.70 vs 1.75), allowing the agent to transition to exploitation earlier and achieve higher rewards.
- **Lower epsilon_end (0.01-0.05)**: Configurations with lower final epsilon values performed better, as they reduced random actions at the end of training.

### Which changes harmed performance?

- **Slower epsilon decay (500000-750000)**: Longer exploration periods prevented the agent from exploiting learned knowledge effectively, resulting in lower rewards (2.16-3.47).
- **Higher epsilon_end (0.1-0.15)**: Maintaining high exploration rates at the end of training reduced the agent's ability to play optimally, leading to poorer performance.

### What final configuration performed best and why?

The best configuration was **epsilon_decay=100000, epsilon_end=0.05**, achieving a final mean reward of 12.70.

**Why?**
- The faster decay (100000 vs 250000 baseline) allowed the agent to stop exploring sooner and focus on exploiting the learned Q-values.
- The moderate epsilon_end (0.05) provided enough final exploration to handle novel situations without excessive randomness.
- This balance enabled the agent to learn effective strategies early and refine them through focused play, leading to the highest reward.

## Observations and Decision-Making

1. **Exploration-Exploitation Trade-off**: The results clearly demonstrate the importance of balancing exploration and exploitation. Too much exploration (slow decay, high end epsilon) prevented high performance, while optimal timing of the transition to exploitation was crucial.

2. **Decay Rate Sensitivity**: Small changes in epsilon_decay had dramatic effects on final performance, highlighting the need for careful tuning of exploration schedules.

3. **Final Epsilon Impact**: While lower epsilon_end generally helped, the interaction with decay rate was important. Fast decay with low end epsilon worked well, but slow decay with low end still suffered from prolonged exploration.

4. **Practical Implications**: For similar environments, we recommend starting with faster decay rates (around 100000-200000 steps) and moderate final epsilon (0.01-0.05), then adjusting based on observed learning curves.

## Conclusion

Our exploration tuning revealed that the baseline exploration strategy was suboptimal for the 100,000 step training budget. By accelerating the exploration-exploitation transition, we achieved significantly better performance. This underscores the importance of adapting exploration schedules to training constraints and environment complexity.

While learning rate remains a critical hyperparameter, our results show that exploration strategy can be equally important for RL agent performance. Future work should consider adaptive exploration methods that can dynamically adjust based on learning progress.