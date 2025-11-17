# DQN Agent for Atari Breakout

This project is a submission for our course in Machine Learning. The objective is to train, evaluate, and analyze a Deep Q-Network (DQN) agent to play the Atari game `ALE/Breakout-v5` using the `gymnasium` and `stable_baselines3` libraries.

This document serves as the complete and final report for the assignment.

## Project Structure

```
/
├── models/           # Stores all trained .zip models
├── logs/             # Stores all TensorBoard log files
├── train.py          # Reusable script for training agents
├── play.py           # Script for playing/watching trained models
├── requirements.txt    # All project dependencies
└── README.md         # This report
```



## Installation & Setup

This project was built using Python `3.11`. A stable version like `3.10` or `3.11` is required.

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/m-mwangi/Formative_3_Group_Work.git
    cd Formative_3_Group_Work
    ```

2.  **Create a Virtual Environment:**

    ```bash
    # On macOS / Linux
    python3 -m venv venv

    # On Windows
    python -m venv venv
    ```

3.  **Activate the Environment:**

    ```bash
    # On macOS / Linux
    source venv/bin/activate

    # On Windows (PowerShell)
    .\venv\Scripts\Activate.ps1
    ```

4.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```



## Usage

All scripts are run from the command line and are configured using arguments.

### 1\. How to Train an Agent

The `train.py` script is the "factory" for all our experiments. It automatically saves the final model to `models/` and the log files to `logs/` using a name based on the hyperparameters.

**Base Command:**

```bash
python train.py --steps [NUMBER_OF_STEPS] [ARGUMENTS]
```

**Key Arguments:**

  * `--steps`: Total training steps (e.g., `250000`).
  * `--lr`: Learning rate (e.g., `0.0005`).
  * `--gamma`: Discount factor (e.g., `0.99`).
  * `--epsilon_decay`: Steps to decay exploration (e.g., `150000`).
  * `--policy`: Policy to use (`CnnPolicy` or `MlpPolicy`).

**Example:**

```bash
python train.py --steps 250000 --lr 0.0005 --epsilon_decay 150000
```

### 2\. How to Watch an Agent Play

The `play.py` script loads a saved model file and renders the game in a new window.

**Command:**

```bash
python play.py --model "PATH_TO_YOUR_MODEL.zip"
```

**Example (Using Tab-Complete):**
The model names are long. Use **Tab** to auto-complete the name.

```bash
# Type this and press Tab:
python play.py --model models/policy-CnnPolicy_lr-0.0005
```

### 3\. How to Evaluate Results

We use **TensorBoard** to read the `logs/` folder and analyze our results.

1.  **Start TensorBoard:**
    ```bash
    tensorboard --logdir logs/
    ```
2.  **Open in Browser:** Open the URL in your terminal (usually `http://localhost:6006/`).
3.  **Analyze:** Find the **`rollout/ep_rew_mean`** graph for each run to find its stable, average score.



## Core Concept: `CnnPolicy` vs. `MlpPolicy`

Before tuning, we confirmed our understanding of the required architecture.

  * **`CnnPolicy` (Convolutional Neural Network):** This policy is designed to read images. It understands 2D relationships, such as the ball's position relative to the paddle.
  * **`MlpPolicy` (Multi-Layer Perceptron):** This policy is for simple vector data (a flat list of numbers). When fed an image, it "flattens" it, losing all spatial information.

Our test (Experiment 4.11) proved this:

  * **`CnnPolicy` (Run 4.1):** Achieved a stable score of **\~10.0**.
  * **`MlpPolicy` (Run 4.11):** Performed significantly worse, scoring only **\~4.97**.

This confirms that a **`CnnPolicy` is essential** for an agent to understand and play `Breakout`.



## Hyperparameter Tuning & Results

We conducted 40 unique experiments, with each of our four group members acting as a " to analyze a specific set of parameters. All experiments were run for **250,000 steps** for a fair comparison.

### Member 1: "Learning Rate & Stability"

  * **Mission:** To find the optimal `learning_rate` (`lr`) and test its interaction with `batch_size`.
  * **Table:**

| Exp  | Key Hyperparameters                | Final Avg. Reward | Analysis / Notes |
|------|-----------------------------------|------------------|------------------|
| 1.1  | lr = 0.001               | 0.5              | Unstable and overshoots gradients. |
| 1.2  | lr = 0.0005, batch = 64  | 3.6              | Best performance so far. Strong learning progress. |
| 1.3  | lr = 0.00025, batch = 64 | 1.6              | Stable but slower improvements. Could improve with more steps. |
| 1.4  | lr = 0.0001, batch = 64  | 1.4              | Stable baseline learning, but underfitting at 500k steps. |
| 1.5  | lr = 0.00005, batch = 64 | 0.2              | Learning rate too small, almost no learning. |
| 1.6  | lr = 0.00001             | 0.1              | Expected to fail learning due to extremely tiny updates. |
| 1.7  | lr = 0.005, batch = 64   | 1.5              | Good performance but still not the best. |
| 1.8  | lr = 0.0001, batch = 128 | 1.2              | Larger batch reduced exploration and slowed learning. |
| 1.9  | lr = 0.0005, batch = 128 | 2.4              | Good performance but slower learning than batch 64. |
| 1.10 | lr = 0.00005, batch = 16 | 0.3              | Small batch helps exploration, but lr is still too low. |


### Member 2: "Exploration & Curiosity"

  * **Mission:** To find the best exploration strategy by tuning `epsilon_decay` and `epsilon_end`.
  * **Table:**

| Exp | Key Hyperparameters | Final Avg. Reward (`ep_rew_mean`) | Analysis / Noted Behavior |
| :-- | :--- | :--- | :--- |
| 2.1 | `decay=100k` | *[Member 2 to fill in]* | *[Member 2: Fill in]* |
| 2.2 | `decay=250k` (Baseline) | *[Member 2 to fill in]* | *[Member 2: Fill in]* |
| 2.3 | `decay=500k` | *[Member 2 to fill in]* | *[Member 2: Fill in]* |
| 2.4 | `decay=750k` | *[Member 2 to fill in]* | *[Member 2: Fill in]* |
| 2.5 | `decay=100k`, `end=0.01` | *[Member 2 to fill in]* | *[Member 2: Fill in]* |
| 2.6 | `decay=500k`, `end=0.1` | *[Member 2 to fill in]* | *[Member 2: Fill in]* |
| 2.7 | `end=0.001` | *[Member 2 to fill in]* | *[Member 2: Fill in]* |
| 2.8 | `end=0.025` | *[Member 2 to fill in]* | *[Member 2: Fill in]* |
| 2.9 | `end=0.1` | *[Member 2 to fill in]* | *[Member 2: Fill in]* |
| 2.10 | `end=0.15` | *[Member 2 to fill in]* | *[Member 2: Fill in]* |

### Member 3: "Long-Term Vision"

  * **Mission:** To find the optimal discount factor (`gamma`) and its interaction with `lr`.
  * **Table:**

| Exp | Key Hyperparameters | Final Avg. Reward (`ep_rew_mean`) | Analysis / Noted Behavior |
| :-- | :--- | :--- | :--- |
| 3.1 | `gamma=0.9` | *[Member 3 to fill in]* | *[Member 3: Fill in]* |
| 3.2 | `gamma=0.95` | *[Member 3 to fill in]* | *[Member 3: Fill in]* |
| 3.3 | `gamma=0.98` | *[Member 3 to fill in]* | *[Member 3: Fill in]* |
| 3.4 | `gamma=0.99` (Baseline) | *[Member 3 to fill in]* | *[Member 3: Fill in]* |
| 3.5 | `gamma=0.995` | *[Member 3 to fill in]* | *[Member 3: Fill in]* |
| 3.6 | `gamma=0.95`, `lr=0.0005` | *[Member 3 to fill in]* | *[Member 3: Fill in]* |
| 3.7 | `gamma=0.99`, `lr=0.0005` | *[Member 3 to fill in]* | *[Member 3: Fill in]* |
| 3.8 | `gamma=0.95`, `lr=0.00005` | *[Member 3 to fill in]* | *[Member 3: Fill in]* |
| 3.9 | `gamma=0.99`, `lr=0.00005` | *[Member 3 to fill in]* | *[Member 3: Fill in]* |
| 3.10 | `gamma=0.9`, `lr=0.0005` | *[Member 3 to fill in]* | *[Member 3: Fill in]* |

### Member 4: "Fine-Tuning" learning rate & epsilon decay

  * **Mission:** To perform a "deep dive" grid search on the most promising `lr` and `epsilon_decay` values.
  * **Table:**

| Exp | Key Hyperparameters | Final Avg. Reward (`ep_rew_mean`) | Analysis / Noted Behavior |
| :-- | :--- | :--- | :--- |
| 4.1 | `lr=0.0001`, `decay=250k` (Baseline) | **\~10.0** | Agent's reward jumped to \~10 as the 250k exploration phase ended. Stable. |
| 4.2 | `lr=0.0005`, `decay=150k` | **\~13.4** | **(Best Result)**. A faster `lr` (0.0005) and faster decay (150k) was stable and performed significantly better than the baseline. |
| 4.3 | `lr=0.0005`, `decay=250k` | **\~12.4** | Good score, but slightly worse than 4.2. Suggests a fast `lr` pairs better with a fast decay. |
| 4.4 | `lr=0.0005`, `decay=400k` | \~5.88 | Poor performance. The slow decay (400k) meant the agent was still exploring (`exploration_rate` = 0.407) when the run ended. |
| 4.5 | `lr=0.00025`, `decay=150k` | **\~11.4** | A very stable, good result. Better than baseline, but not as high as the faster `lr` (0.0005). |
| 4.6 | `lr=0.00025`, `decay=250k` | **\~11.4** | Identical to 4.5. Proves that for a "mid" `lr`, the decay speed (150k vs 250k) had no significant impact. |
| 4.7 | `lr=0.00025`, `decay=400k` | \~4.98 | Poor performance, as predicted. Agent was still exploring (`exploration_rate` = 0.406) when the run ended. |
| 4.8 | `lr=0.0001`, `decay=150k` | **\~10.3** | Stable run, almost identical to the baseline (4.1). |
| 4.9 | `lr=0.0001`, `decay=400k` | \~5.19 | Poor performance. Confirms a slow decay (400k) is a bad strategy for a 250k step run. |
| 4.10 | `lr=0.0005`, `decay=150k`, `gamma=0.98` | **\~13.4** | **(Tied for Best)**. Identical to 4.2. Proves that a small change in `gamma` (0.99 vs 0.98) had no impact on this optimal setup. |
| 4.11 | `policy=MlpPolicy` | \~4.97 | **Proof of Concept.** Confirmed `MlpPolicy` is the wrong choice. It ran but scored 50% worse than the `CnnPolicy` baseline. |



## Analysis & Key Findings

From our combined 40 experiments, we will draw several conclusions. Based on the "Fine-Tuning" (Member 4) results, we have these key findings:

1.  **The Learning Rate is a Dominant Factor:** A higher `lr` of `0.0005` (Run 4.2) performed significantly better than the baseline `0.0001` (Run 4.1), yielding a \~34% increase in average reward.
2.  **Exploration Must Finish:** The `epsilon_decay` parameter must be set low enough to finish within the `total_steps`. In all runs where decay was set to 400k (e.g., 4.4, 4.7, 4.9), the agent was still exploring at the end, and performance was poor (scores of \~5).
3.  **Optimal Pairing:** The best-performing combination was a **fast learning rate (`lr=0.0005`)** paired with a **fast exploration decay (`decay=150k`)**. This allowed the agent to learn quickly and then have a long "exploitation" phase (100,000 steps) to use its knowledge.
4.  **`Gamma` is Not Sensitive (at high values):** The difference between `gamma=0.99` (Run 4.2) and `gamma=0.98` (Run 4.10) was negligible, with both achieving an identical score of \~13.4.



## Final Model & Demonstration

Based on our analysis, the best-performing model from all 40 experiments will be selected for the final demonstration.

  * **Best Model:** `models/policy-CnnPolicy_lr-0.0005_gamma-0.99_batch-32_eps_start-1.0_eps_end-0.05_eps_decay-150000.zip`
  * **Demonstration Video:** `[Video Demo](link to the video)`


## Group Contributions

  * **Member 1:** `Christian Iradukunda Byiringiro`
  * **Member 2:** `Marion Mwangi`
  * **Member 3:** `Irenee Dusingizimana`
  * **Member 4:** `Christope Gakwaya`
