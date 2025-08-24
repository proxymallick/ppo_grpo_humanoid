import datetime
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from lib.utils_grpo import parse_args_grpo, make_env, log_video
from lib.agent_grpo import GRPOAgent
from lib.buffer_grpo import GRPOBuffer
def grpo_update(agent, optimizer, scaler, batch_obs, batch_actions, batch_returns, batch_old_log_probs, batch_adv,
               batch_reconst, batch_latent, clip_epsilon, vf_coef, ent_coef, reconst_coef, kl_coef):
    agent.train()

    optimizer.zero_grad()
    
    # Disable autocast for the forward pass to avoid half-precision issues with distributions
    with torch.cuda.amp.autocast(enabled=False):
        # Get the new log probabilities, entropies, values, reconstructions and latents
        _, new_log_probs, entropies, new_values, new_reconst, new_latent = agent.get_action_and_value(batch_obs, batch_actions)
        ratio = torch.exp(new_log_probs - batch_old_log_probs)

        # Normalize the advantages
        batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)

        # Calculate the policy loss (PPO part)
        surr1 = ratio * batch_adv
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_adv
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(new_values.squeeze(1), batch_returns)
        
        # Entropy term
        entropy = entropies.mean()
        
        # Reconstruction loss (generative component)
        reconstruction_loss = nn.MSELoss()(new_reconst, batch_obs)
        
        # KL divergence to regularize the latent space (optional)
        # Assuming a prior of N(0, 1) for the latent space
        latent_kl_loss = 0.5 * torch.mean(
            torch.sum(new_latent ** 2 - batch_latent ** 2 - 1 + 2 * (batch_latent - new_latent) ** 2, dim=-1)
        )
        
        # Total loss
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy + reconst_coef * reconstruction_loss + kl_coef * latent_kl_loss

    # Manually do the backward pass with scaled gradients
    scaler.scale(loss).backward()
    
    # Make sure we don't have any NaN gradients
    for param in agent.parameters():
        if param.grad is not None:
            torch.nan_to_num_(param.grad, nan=0.0, posinf=1.0, neginf=-1.0)
    
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)  # Reduced from 1.0 for more stability
    scaler.step(optimizer)
    scaler.update()

    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item(), reconstruction_loss.item(), latent_kl_loss.item()

if __name__ == "__main__":
    args = parse_args_grpo()
    device = torch.device("cuda" if args.cuda else "cpu")

    # Create the folders for logging
    current_dir = os.path.dirname(__file__)
    folder_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    videos_dir = os.path.join(current_dir, "videos", folder_name)
    os.makedirs(videos_dir, exist_ok=True)
    checkpoint_dir = os.path.join(current_dir, "checkpoints", folder_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create the tensorboard writer
    log_dir = os.path.join(current_dir, "logs", folder_name)
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Create the environments
    envs = gym.vector.AsyncVectorEnv(
        [lambda: make_env(args.env, reward_scaling=args.reward_scale) for _ in range(args.n_envs)])
    test_env = make_env(args.env, reward_scaling=args.reward_scale, render=True)
    obs_dim = envs.single_observation_space.shape
    act_dim = envs.single_action_space.shape

    # Create the agent and optimizer
    agent = GRPOAgent(obs_dim[0], act_dim[0], latent_dim=args.latent_dim).to(device)

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

    # Create the buffer
    buffer = GRPOBuffer(obs_dim, act_dim, args.n_steps, args.n_envs, device, args.gamma, args.gae_lambda, args.latent_dim)

    # Start the training
    start_time = time.time()
    next_obs = torch.tensor(np.array(envs.reset()[0], dtype=np.float32), device=device)
    next_terminateds = torch.zeros(args.n_envs, dtype=torch.float32, device=device)
    next_truncateds = torch.zeros(args.n_envs, dtype=torch.float32, device=device)
    reward_list = []

    try:
        for epoch in range(start_epoch, args.n_epochs + 1):

            # Collect trajectories
            for _ in tqdm(range(0, args.n_steps), desc=f"Epoch {epoch}: Collecting trajectories"):
                global_step_idx += args.n_envs
                obs = next_obs
                terminateds = next_terminateds
                truncateds = next_truncateds

                # Sample the actions
                with torch.no_grad():
                    actions, logprobs, _, values, reconstructions, latents = agent.get_action_and_value(obs)
                    values = values.reshape(-1)

                # Step the environment
                next_obs, rewards, next_terminateds, next_truncateds, _ = envs.step(actions.cpu().numpy())

                # parse everything to tensors
                next_obs = torch.tensor(np.array(next_obs, dtype=np.float32), device=device)
                reward_list.extend(rewards)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_terminateds = torch.as_tensor(next_terminateds, dtype=torch.float32, device=device)
                next_truncateds = torch.as_tensor(next_truncateds, dtype=torch.float32, device=device)

                # Store the step in the buffer
                buffer.store(obs, actions, rewards, values, terminateds, truncateds, logprobs, reconstructions, latents)

            # After the trajectories are collected, calculate the advantages and returns
            with torch.no_grad():
                # Finish the last step of the buffer with the value of the last state
                # and the terminated and truncated flags
                next_values = agent.get_value(next_obs).reshape(1, -1)
                next_terminateds = next_terminateds.reshape(1, -1)
                next_truncateds = next_truncateds.reshape(1, -1)
                traj_adv, traj_ret = buffer.calculate_advantages(next_values, next_terminateds, next_truncateds)

            # Get the stored trajectories from the buffer
            traj_obs, traj_act, traj_logprob, traj_reconst, traj_latent = buffer.get()

            # Flatten the trajectories
            traj_obs = traj_obs.view(-1, *obs_dim)
            traj_act = traj_act.view(-1, *act_dim)
            traj_logprob = traj_logprob.view(-1)
            traj_adv = traj_adv.view(-1)
            traj_ret = traj_ret.view(-1)
            traj_reconst = traj_reconst.view(-1, *obs_dim)
            traj_latent = traj_latent.view(-1, args.latent_dim)

            # Create an array of indices to sample from the trajectories
            dataset_size = args.n_steps * args.n_envs
            traj_indices = np.arange(dataset_size)

            sum_loss_policy = 0.0
            sum_loss_value = 0.0
            sum_entropy = 0.0
            sum_loss_reconst = 0.0
            sum_loss_kl = 0.0
            sum_loss_total = 0.0
            
            for _ in tqdm(range(args.train_iters), desc=f"Epoch {epoch}: Training"):
                # Shuffle the indices
                np.random.shuffle(traj_indices)
                # Iterate over the batches
                for start_idx in range(0, dataset_size, args.batch_size):
                    end_idx = start_idx + args.batch_size
                    batch_indices = traj_indices[start_idx:end_idx]

                    batch_obs = traj_obs[batch_indices]
                    batch_actions = traj_act[batch_indices]
                    batch_returns = traj_ret[batch_indices]
                    batch_old_log_probs = traj_logprob[batch_indices]
                    batch_adv = traj_adv[batch_indices]
                    batch_reconst = traj_reconst[batch_indices]
                    batch_latent = traj_latent[batch_indices]

                    loss, policy_loss, value_loss, entropy, reconst_loss, kl_loss = grpo_update(
                        agent, optimizer, scaler, batch_obs, batch_actions, batch_returns,
                        batch_old_log_probs, batch_adv, batch_reconst, batch_latent,
                        args.clip_ratio, args.vf_coef, args.ent_coef, args.reconst_coef, args.kl_coef
                    )

                    sum_loss_policy += policy_loss
                    sum_loss_value += value_loss
                    sum_entropy += entropy
                    sum_loss_reconst += reconst_loss
                    sum_loss_kl += kl_loss
                    sum_loss_total += loss

            # Log the losses
            total_loss = sum_loss_total / args.train_iters / (dataset_size / args.batch_size)
            policy_loss = sum_loss_policy / args.train_iters / (dataset_size / args.batch_size)
            value_loss = sum_loss_value / args.train_iters / (dataset_size / args.batch_size)
            entropy = sum_entropy / args.train_iters / (dataset_size / args.batch_size)
            reconst_loss = sum_loss_reconst / args.train_iters / (dataset_size / args.batch_size)
            kl_loss = sum_loss_kl / args.train_iters / (dataset_size / args.batch_size)
            
            writer.add_scalar("loss/total", total_loss, epoch)
            writer.add_scalar("loss/policy", policy_loss, epoch)
            writer.add_scalar("loss/value", value_loss, epoch)
            writer.add_scalar("loss/entropy", entropy, epoch)
            writer.add_scalar("loss/reconstruction", reconst_loss, epoch)
            writer.add_scalar("loss/kl", kl_loss, epoch)

            # Log learning rate
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch)

            # Log the rewards
            mean_reward = float(np.mean(reward_list) / args.reward_scale)
            writer.add_scalar("reward/mean", mean_reward, epoch)
            reward_list = []
            print(f"Epoch {epoch} done in {time.time() - start_time:.2f}s, mean reward: {mean_reward:.2f}, "
                  f"total loss: {total_loss:.4f}, policy loss: {policy_loss:.4f}, value loss: {value_loss:.4f}, "
                  f"entropy: {entropy:.4f}, reconst loss: {reconst_loss:.4f}, kl loss: {kl_loss:.4f}, "
                  f"learning rate: {scheduler.get_last_lr()[0]:.2e}")
            start_time = time.time()

            # Save the model if the mean reward is better
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Save with the new complete checkpoint format
                torch.save({
                    'model_state_dict': agent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step_idx,
                    'best_mean_reward': best_mean_reward
                }, os.path.join(checkpoint_dir, "best.pt"))
                print(f"New best model saved with mean reward: {mean_reward:.2f}")

            # Save the last model
            torch.save({
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch,
                'global_step': global_step_idx,
                'best_mean_reward': best_mean_reward
            }, os.path.join(checkpoint_dir, "last.pt"))

            # Every n epochs, log the video
            if epoch % args.render_epoch == 0:
                log_video(test_env, agent, device, os.path.join(videos_dir, f"epoch_{epoch}.mp4"))

            # Update the learning rate
            scheduler.step()

    finally:
        # Close the environments and tensorboard writer
        envs.close()
        test_env.close()
        writer.close()