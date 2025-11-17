import argparse
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import ProgressBarCallback
import os

# Define the environment ID
ENV_ID = "ALE/Breakout-v5"

# Define base directories
MODEL_DIR = "models"
LOG_DIR = "logs"

def parse_args():
    """
    Parses the command line arguments for training.
    """
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari Breakout")
    
    # Model Hyperparameters
    parser.add_argument('--policy', type=str, default="CnnPolicy",
                        help='The policy to use (CnnPolicy or MlpPolicy)')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate for the optimizer (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.99, 
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training (default: 32)')
    
    # Epsilon-Greedy Parameters
    parser.add_argument('--epsilon_start', type=float, default=1.0, 
                        help='Starting value of epsilon (default: 1.0)')
    parser.add_argument('--epsilon_end', type=float, default=0.05, 
                        help='Minimum value of epsilon (default: 0.05)')
    parser.add_argument('--epsilon_decay', type=float, default=250000, 
                        help='Number of timesteps to decay epsilon (default: 250,000)')

    # Training Parameters
    parser.add_argument('--steps', type=int, default=1000000, 
                        help='Total number of training timesteps (default: 1,000,000)')
    parser.add_argument('--n_envs', type=int, default=4, 
                        help='Number of parallel environments to run (default: 4)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Naming Logic
    run_name = (
        f"policy-{args.policy}_lr-{args.lr}_gamma-{args.gamma}_"
        f"batch-{args.batch_size}_eps_start-{args.epsilon_start}_" 
        f"eps_end-{args.epsilon_end}_eps_decay-{int(args.epsilon_decay)}" 
    )
    
    # Create the full path for the model file
    model_save_path = os.path.join(MODEL_DIR, f"{run_name}.zip")

    # Create the full path for the log directory
    log_save_path = os.path.join(LOG_DIR, run_name)
    # End of Naming Logic

    print(f"****** Starting Training ******")
    print(f"Environment: {ENV_ID}")
    print(f"Policy: {args.policy}")
    print(f"Will save model to: {model_save_path}")
    print(f"Will save logs to: {log_save_path}")
    print(f"Arguments: {vars(args)}")
    
    # Create the vectorized environment
    env = make_atari_env(ENV_ID, n_envs=args.n_envs)
    
    # Apply the Frame Stack wrapper
    env = VecFrameStack(env, n_stack=4)

    # Define the DQN model
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

    # Train the model
    print(f"\nStarting training for {args.steps} timesteps...")
    progress_bar_callback = ProgressBarCallback()
    model.learn(
        total_timesteps=args.steps,
        log_interval=10,
        callback=progress_bar_callback
    )
    
    # Save the final model
    model.save(model_save_path)
    
    print("***** Training Complete *****")
    print(f"Model saved to: {model_save_path}")

    env.close()

if __name__ == "__main__":
    main()