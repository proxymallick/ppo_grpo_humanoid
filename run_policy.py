import os
import time
import argparse
import numpy as np
import torch
import gymnasium as gym

#from lib.agent_ppo import PPOAgent
from lib.agent_grpo import GRPOAgent
from lib.utils import make_env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/home/prakash/Downloads/PPO-Humanoid/checkpoints/2025-04-05_10-36-52/best.pt', help="Path to the trained model (.pt file)")
    parser.add_argument("--env", type=str, default="Humanoid-v4", help="Gymnasium environment name")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--reward_scale", type=float, default=1.0, help="Reward scaling factor")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--record", action="store_true", help="Record video")
    parser.add_argument("--video_path", type=str, default="videos/evaluation", help="Path to save videos")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    return parser.parse_args()

def run_model(args):
    # Set up device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = make_env(args.env, reward_scaling=args.reward_scale, render=args.render)
    
    # Get observation and action dimensions
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    
    print(f"Observation space: {obs_dim}, Action space: {act_dim}")
    
    # Create agent
    agent = GRPOAgent(obs_dim[0], act_dim[0]).to(device)
    
    # Load model weights
    agent.load_state_dict(torch.load(args.model_path, map_location=device))
    agent.eval()
    print(f"Model loaded from {args.model_path}")
    
    # Create video directory if recording
    if args.record:
        os.makedirs(args.video_path, exist_ok=True)
        env = gym.wrappers.RecordVideo(env, args.video_path)
    
    # Run evaluation episodes
    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0
        
        while not (done or truncated):
            # Convert observation to tensor
            obs_tensor = torch.tensor(np.array(obs, dtype=np.float32), device=device).unsqueeze(0)
            
            # Get action from model
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
                action = action.cpu().numpy().flatten()
            
            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            step += 1
            
            if args.render:
                env.render()
                time.sleep(0.01)  # Small delay to allow for rendering
        
        print(f"Episode {episode+1}/{args.episodes} - Steps: {step}, Reward: {total_reward}")
    
    env.close()
    print("Evaluation completed!")

if __name__ == "__main__":
    args = parse_args()
    run_model(args)