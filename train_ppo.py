import os
import time
import gymnasium as gym
import numpy as np
import torch
import datetime
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from lib.utils import parse_args_ppo, make_env, log_video
from lib.agent_ppo import PPOAgent
from lib.buffer_ppo import PPOBuffer

def ppo_update(agent, optimizer, scaler, batch_obs: torch.Tensor, batch_actions: torch.Tensor,
               batch_returns: torch.Tensor, batch_old_log_probs: torch.Tensor, batch_adv: torch.Tensor,
               clip_epsilon: float, vf_coef: float, ent_coef: float):
    agent.train()  # Set training mode
    optimizer.zero_grad()
    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
        _, new_log_probs, entropies, new_values = agent.get_action_and_value(batch_obs, batch_actions)
        ratio = torch.exp(new_log_probs - batch_old_log_probs)
        surr1 = ratio * batch_adv
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_adv
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(new_values.squeeze(1), batch_returns)
        entropy = entropies.mean()
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()

if __name__ == "__main__":
    args = parse_args_ppo()
    device = torch.device("cuda" if args.cuda else "cpu")

    # Create logging folders
    current_dir = os.path.dirname(__file__)
    folder_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    videos_dir = os.path.join(current_dir, "videos", folder_name)
    os.makedirs(videos_dir, exist_ok=True)
    checkpoint_dir = os.path.join(current_dir, "checkpoints", folder_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Tensorboard writer
    log_dir = os.path.join(current_dir, "logs", folder_name)
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Create environments
    def create_env():
        return make_env(args.env, reward_scaling=args.reward_scale)

    env_fns = [create_env for _ in range(args.n_envs)]
    envs = gym.vector.AsyncVectorEnv(env_fns)
    test_env = make_env(args.env, reward_scaling=args.reward_scale, render=True)
    obs_dim = envs.single_observation_space.shape
    act_dim = envs.single_action_space.shape

    # Create agent and optimizer
    agent = PPOAgent(obs_dim[0], act_dim[0]).to(device)

    


    # Update the learning rate and warmup parameters for more stability
    args.learning_rate = 0.0001  # Reduced from the original value
    warmup_epochs = 5            # Shorter warmup

    # Create the optimizer with lower epsilon value
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-8)  # Changed from 1e-5

    # Create a more gradual learning rate scheduler
    def lr_lambda(epoch):
        # Linear warmup for the first 'warmup_epochs' epochs
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            # More gradual decay after the warmup epochs
            return 0.9995 ** (epoch - warmup_epochs)  # Changed from 0.999

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Update the gradient scaling parameters for mixed precision training
    scaler = torch.amp.GradScaler(enabled=str(device) == "cuda", init_scale=2**10)  # Explicit init_scale

    # Initialize variables
    global_step_idx = 0
    best_mean_reward = -np.inf
    start_epoch = 1

    # Resuming training if specified
    if args.resume:
        if args.checkpoint_path is None:
            print("Error: --checkpoint_path must be specified when --resume is used")
            exit(1)
        
        checkpoint_path = args.checkpoint_path
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file {checkpoint_path} not found")
            exit(1)
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle both old and new checkpoint format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format with full state
            agent.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            
            if 'global_step' in checkpoint:
                global_step_idx = checkpoint['global_step']
            
            if 'best_mean_reward' in checkpoint:
                best_mean_reward = checkpoint['best_mean_reward']
        else:
            # Old format with just model state dict
            agent.load_state_dict(checkpoint)
            print("Using old checkpoint format. Only model weights loaded.")
            
        # Override epoch if specified in command line
        if args.resume_epoch is not None:
            start_epoch = args.resume_epoch
            print(f"Overriding start epoch to {start_epoch} as specified")
        
        # If we want to specifically start at epoch 151
        if start_epoch != 151 and args.resume_epoch is None:
            start_epoch = 151
            print("Defaulting to start at epoch 151")
        
        print(f"Resuming training from epoch {start_epoch}")
        print(f"Current best mean reward: {best_mean_reward}")

    print(agent)

    # Create buffer
    buffer = PPOBuffer(obs_dim, act_dim, args.n_steps, args.n_envs, device, args.gamma, args.gae_lambda)

    # Start training
    global_step_idx = 0
    best_mean_reward = -np.inf
    start_time = time.time()
    next_obs = torch.tensor(np.array(envs.reset()[0], dtype=np.float32), device=device)
    next_terminateds = torch.zeros(args.n_envs, dtype=torch.float32, device=device)
    next_truncateds = torch.zeros(args.n_envs, dtype=torch.float32, device=device)
    reward_list = []
    patience = 10
    epochs_without_improvement = 0

    try:    
        for epoch in range(start_epoch, args.n_epochs + 1):
            # Collect trajectories
            for _ in tqdm(range(0, args.n_steps), desc=f"Epoch {epoch}: Collecting trajectories"):
                global_step_idx += args.n_envs
                obs = next_obs
                terminateds = next_terminateds
                truncateds = next_truncateds

                # Sample actions
                with torch.no_grad():
                    actions, logprobs, _, values = agent.get_action_and_value(obs)
                    values = values.reshape(-1)

                # Step environment
                next_obs, rewards, next_terminateds, next_truncateds, _ = envs.step(actions.cpu().numpy())

                # Parse to tensors
                next_obs = torch.tensor(np.array(next_obs, dtype=np.float32), device=device)
                reward_list.extend(rewards)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_terminateds = torch.as_tensor(next_terminateds, dtype=torch.float32, device=device)
                next_truncateds = torch.as_tensor(next_truncateds, dtype=torch.float32, device=device)

                # Store step
                buffer.store(obs, actions, rewards, values, terminateds, truncateds, logprobs)

            # Calculate advantages and returns
            with torch.no_grad():
                next_values = agent.get_value(next_obs).reshape(1, -1)
                next_terminateds = next_terminateds.reshape(1, -1)
                next_truncateds = next_truncateds.reshape(1, -1)
                traj_adv, traj_ret = buffer.calculate_advantages(next_values, next_terminateds, next_truncateds)

            # Get trajectories
            traj_obs, traj_act, traj_logprob = buffer.get()
            traj_obs = traj_obs.view(-1, *obs_dim)
            traj_act = traj_act.view(-1, *act_dim)
            traj_logprob = traj_logprob.view(-1)
            traj_adv = traj_adv.view(-1)
            traj_ret = traj_ret.view(-1)

            # Normalize advantages
            traj_adv = (traj_adv - traj_adv.mean()) / (traj_adv.std() + 1e-8)

            # Training
            dataset_size = args.n_steps * args.n_envs
            traj_indices = np.arange(dataset_size)
            sum_loss_policy = 0.0
            sum_loss_value = 0.0
            sum_entropy = 0.0
            sum_loss_total = 0.0
            for _ in tqdm(range(args.train_iters), desc=f"Epoch {epoch}: Training"):
                np.random.shuffle(traj_indices)
                for start_idx in range(0, dataset_size, args.batch_size):
                    end_idx = start_idx + args.batch_size
                    batch_indices = traj_indices[start_idx:end_idx]

                    batch_obs = traj_obs[batch_indices]
                    batch_actions = traj_act[batch_indices]
                    batch_returns = traj_ret[batch_indices]
                    batch_old_log_probs = traj_logprob[batch_indices]
                    batch_adv = traj_adv[batch_indices]

                    loss, policy_loss, value_loss, entropy = ppo_update(
                        agent, optimizer, scaler, batch_obs, batch_actions, batch_returns,
                        batch_old_log_probs, batch_adv, args.clip_ratio, args.vf_coef, args.ent_coef
                    )

                    sum_loss_policy += policy_loss
                    sum_loss_value += value_loss
                    sum_entropy += entropy
                    sum_loss_total += loss

            # Log losses
            total_loss = sum_loss_total / args.train_iters / (dataset_size / args.batch_size)
            policy_loss = sum_loss_policy / args.train_iters / (dataset_size / args.batch_size)
            value_loss = sum_loss_value / args.train_iters / (dataset_size / args.batch_size)
            entropy = sum_entropy / args.train_iters / (dataset_size / args.batch_size)
            writer.add_scalar("loss/total", total_loss, epoch)
            writer.add_scalar("loss/policy", policy_loss, epoch)
            writer.add_scalar("loss/value", value_loss, epoch)
            writer.add_scalar("loss/entropy", entropy, epoch)

            # Log learning rate
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch)

            # Log rewards
            mean_reward = float(np.mean(reward_list) / args.reward_scale)
            writer.add_scalar("reward/mean", mean_reward, epoch)
            reward_list = []
            print(f"Epoch {epoch} done in {time.time() - start_time:.2f}s, mean reward: {mean_reward:.2f}, "
                  f"total loss: {total_loss:.4f}, policy loss: {policy_loss:.4f}, value loss: {value_loss:.4f}, "
                  f"entropy: {entropy:.4f}, learning rate: {scheduler.get_last_lr()[0]:.2e}")
            start_time = time.time()

            # Save model if better
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                torch.save(agent.state_dict(), os.path.join(checkpoint_dir, "best.pt"))
                print(f"New best model saved with mean reward: {mean_reward:.2f}")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Save last model
            torch.save(agent.state_dict(), os.path.join(checkpoint_dir, "last.pt"))

            # Log video
            if epoch % args.render_epoch == 0:
                log_video(test_env, agent, device, os.path.join(videos_dir, f"epoch_{epoch}.mp4"))

            # Early stopping
            # if epochs_without_improvement >= patience:
            #     print(f"Early stopping at epoch {epoch}")
            #     break

            # Update learning rate
            scheduler.step()

    finally:
        envs.close()
        test_env.close()
        writer.close()