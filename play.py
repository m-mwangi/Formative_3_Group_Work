import argparse
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
import time

# Define the environment ID
ENV_ID = "ALE/Breakout-v5"

def parse_args():
    """
    Parses the command line arguments for playing.
    """
    parser = argparse.ArgumentParser(description="Play with a trained DQN agent")
    
    # Required Argument
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to the trained model .zip file')
    
    # Optional Arguments
    parser.add_argument('--n_episodes', type=int, default=5, 
                        help='Number of episodes to play (default: 5)')
    
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"***** Loading Model and Playing *****")
    print(f"Loading model from: {args.model}")

    # Create the environment and enforce n_envs=1 for rendering
    env = make_atari_env(ENV_ID, n_envs=1, env_kwargs={"render_mode": "human"})
    env = VecFrameStack(env, n_stack=4)

    # Load the trained model
    try:
        model = DQN.load(args.model, env=env)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure the model path is correct and was trained on this environment.")
        return

    print("Model loaded successfully.")

    # Play the game
    for episode in range(args.n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        print(f"\n***** Starting Episode {episode + 1} *****")
        
        while not done:
            # Get the agent's action
            action, _states = model.predict(obs, deterministic=True)
            
            # Perform the action in the environment
            obs, reward, done, info = env.step(action)
            
            # Render the game
            env.render()
            
            # Add a small delay to be able to watch it
            time.sleep(1/60) # 60 FPS
            
            total_reward += reward[0] # Reward is a vector
            
            # "done" will be True if the episode or game is over
            if done:
                print(f"Episode {episode + 1} finished.")
                print(f"Total Reward: {total_reward}")

    env.close()
    print("\n***** Playback Complete *****")

if __name__ == "__main__":
    main()