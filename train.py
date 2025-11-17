import argparse
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.evaluation import evaluate_policy
import os
import csv

# Define the environment ID
ENV_ID = "ALE/Breakout-v5"

# Define base directories
MODEL_DIR = "models"
LOG_DIR = "logs"
RESULTS_CSV = "results.csv"  # CSV file to store results


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari Breakout")
    
    parser.add_argument('--policy', type=str, default="CnnPolicy")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=32)
    
    parser.add_argument('--epsilon_start', type=float, default=1.0) 
    parser.add_argument('--epsilon_end', type=float, default=0.05)
    parser.add_argument('--epsilon_decay', type=float, default=250000)

    parser.add_argument('--steps', type=int, default=1000000)
    parser.add_argument('--n_envs', type=int, default=4)
    
    return parser.parse_args()


def append_csv(result_row):
    """Append results to results.csv"""
    file_exists = os.path.isfile(RESULTS_CSV)

    with open(RESULTS_CSV, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "run_name", "gamma", "lr", "batch_size",
                "steps", "final_mean_reward"
            ])

        writer.writerow(result_row)


def main():
    args = parse_args()
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Run name for logs + model
    run_name = (
        f"policy-{args.policy}_lr-{args.lr}_gamma-{args.gamma}_"
        f"batch-{args.batch_size}_eps_start-{args.epsilon_start}_"
        f"eps_end-{args.epsilon_end}_eps_decay-{int(args.epsilon_decay)}"
    )
    
    model_save_path = os.path.join(MODEL_DIR, f"{run_name}.zip")
    log_save_path = os.path.join(LOG_DIR, run_name)

    print("****** Starting Training ******")
    print(f"Environment: {ENV_ID}")
    print(f"Saving model to: {model_save_path}")
    print(f"Saving logs to: {log_save_path}")

    # Create vectorized environment
    env = make_atari_env(ENV_ID, n_envs=args.n_envs)
    env = VecFrameStack(env, n_stack=4)

    # DQN Model
    model = DQN(
        args.policy,
        env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        exploration_initial_eps=args.epsilon_start,
        exploration_final_eps=args.epsilon_end,
        exploration_fraction=args.epsilon_decay / args.steps,
        buffer_size=100_000,
        learning_starts=10_000,
        target_update_interval=1_000,
        train_freq=(4, "step"),
        gradient_steps=1,
        verbose=1,
        tensorboard_log=log_save_path
    )

    # Training
    print(f"\nTraining for {args.steps} timesteps...")
    progress_bar_callback = ProgressBarCallback()
    model.learn(
        total_timesteps=args.steps,
        log_interval=10,
        callback=progress_bar_callback
    )

    # Save the model
    model.save(model_save_path)

    # ------------------------------------------
    # ⭐ NEW: Evaluate policy for final reward
    # ------------------------------------------
    print("\nEvaluating final policy (5 episodes)...")

    eval_env = make_atari_env(ENV_ID, n_envs=1)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=5, deterministic=True
    )

    eval_env.close()

    print(f"Final Mean Reward: {mean_reward}")

    # Save results to CSV
    append_csv([
        run_name,
        args.gamma,
        args.lr,
        args.batch_size,
        args.steps,
        mean_reward
    ])

    env.close()

    print("\n***** Training Complete *****")
    print(f"Model saved to: {model_save_path}")
    print(f"Results written to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
