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
from geomloss import SamplesLoss  # For Sinkhorn divergence

from lib.utils_grpo import parse_args_grpo, make_env, log_video
from lib.agent_grpo_full import GRPOAgent
from lib.buffer_grpo_full import GRPOBuffer


def grpo_update(agent, optimizer, scaler, batch_obs: torch.Tensor, batch_actions: torch.Tensor,
                batch_returns: torch.Tensor, batch_old_log_probs: torch.Tensor, batch_adv: torch.Tensor,
                batch_mu_old: torch.Tensor, batch_reconst: torch.Tensor, batch_latent: torch.Tensor,
                old_actor_logstd: torch.Tensor, clip_epsilon: float, vf_coef: float, ent_coef: float,
                reconst_coef: float, kl_coef: float, beta: float, alpha: float, n_groups=4, group_weight=0.5):
    agent.train()  # Set training mode
    optimizer.zero_grad()

    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
        # Get new log probabilities, entropies, values, mu, reconstructions, and latents
        _, new_log_probs, entropies, new_values, mu_theta, new_reconst, new_latent = agent.get_action_and_value(batch_obs, batch_actions)
        ratio = torch.exp(new_log_probs - batch_old_log_probs)

        # Normalize the advantages
        batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)

        # Standard PPO objective
        surr1 = ratio * batch_adv
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_adv
        standard_policy_loss = -torch.min(surr1, surr2)

        # Group Relative Policy Optimization
        batch_size = batch_adv.shape[0]
        group_size = batch_size // n_groups

        # Initialize group losses
        group_losses = []

        # Process each group
        for i in range(n_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size if i < n_groups - 1 else batch_size

            # Get group data
            group_ratio = ratio[start_idx:end_idx]
            group_adv = batch_adv[start_idx:end_idx]

            # Calculate group's performance
            group_perf = (group_ratio * group_adv).mean()

            # Calculate relative performance against other groups
            other_groups_perf = 0
            count = 0
            for j in range(n_groups):
                if j != i:
                    other_start = j * group_size
                    other_end = other_start + group_size if j < n_groups - 1 else batch_size
                    other_groups_perf += (ratio[other_start:other_end] * batch_adv[other_start:other_end]).mean()
                    count += 1

            if count > 0:
                other_groups_perf /= count

            # Relative performance difference
            rel_perf_diff = group_perf - other_groups_perf

            # Loss for this group (minimize negative performance difference)
            group_loss = -rel_perf_diff
            group_losses.append(group_loss)

        # Combine group losses
        group_policy_loss = torch.stack(group_losses).mean()

        # Combined policy loss (standard PPO + group relative component)
        policy_loss = (1 - group_weight) * standard_policy_loss.mean() + group_weight * group_policy_loss

        # Value loss
        value_loss = nn.MSELoss()(new_values.squeeze(1), batch_returns)

        # Entropy term
        entropy = entropies.mean()

        # Reconstruction loss (generative component)
        reconstruction_loss = nn.MSELoss()(new_reconst, batch_obs)

        # KL divergence to regularize the latent space
        latent_kl_loss = 0.5 * torch.mean(
            torch.sum(new_latent ** 2 - batch_latent ** 2 - 1 + 2 * (batch_latent - new_latent) ** 2, dim=-1)
        )

        # KL divergence term for policy
        mu_theta_old = batch_mu_old
        log_std_theta = agent.actor_logstd.expand_as(mu_theta)
        log_std_theta_old = old_actor_logstd.expand_as(mu_theta_old)
        std_theta = torch.exp(log_std_theta)
        std_theta_old = torch.exp(log_std_theta_old)
        
        ratio_log_std = log_std_theta_old - log_std_theta
        term2 = (std_theta ** 2) / (2 * std_theta_old ** 2)
        term3 = ((mu_theta - mu_theta_old) ** 2) / (2 * std_theta_old ** 2)
        KL_per_dim = ratio_log_std + term2 + term3 - 0.5
        KL_per_sample = KL_per_dim.sum(dim=1)
        average_KL = KL_per_sample.mean()

        # Sinkhorn divergence term
        sinkhorn_div = torch.tensor(0.0, device=batch_obs.device)  # Default to 0 if computation fails
        try:
            dist_theta = torch.distributions.Normal(mu_theta, std_theta)
            dist_theta_old = torch.distributions.Normal(mu_theta_old, std_theta_old)
            samples_theta = dist_theta.sample((100,))  # Sample 100 actions
            samples_theta_old = dist_theta_old.sample((100,))
            
            samples_theta_flat = samples_theta.view(-1, samples_theta.shape[-1])
            samples_theta_old_flat = samples_theta_old.view(-1, samples_theta_old.shape[-1])
            
            sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.8)
            sinkhorn_div = sinkhorn(samples_theta_flat, samples_theta_old_flat).mean()

        except Exception as e:
            #import pdb
            #pdb.set_trace()
            #print(f"Warning: Sinkhorn divergence computation failed: {e}")
            sinkhorn_div = torch.tensor(0.0, device=batch_obs.device)  # Fallback to 0

        # Total loss
        loss = (policy_loss + vf_coef * value_loss - ent_coef * entropy +
                reconst_coef * reconstruction_loss + kl_coef * latent_kl_loss -
                beta * average_KL - alpha * sinkhorn_div)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return (loss.item(), policy_loss.item(), value_loss.item(), entropy.item(),
            reconstruction_loss.item(), latent_kl_loss.item(), sinkhorn_div.item())


if __name__ == "__main__":
    args = parse_args_grpo()
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

    # Create agent
    agent = GRPOAgent(obs_dim[0], act_dim[0], latent_dim=args.latent_dim).to(device)

    # Add GRPO-specific parameters
    n_groups = 4  # Number of groups for GRPO
    group_weight = 0.5  # Weight for group relative loss
    args.beta = 0.01  # Weight for KL divergence penalty
    args.alpha = 0.01  # Weight for Sinkhorn divergence penalty
    args.reconst_coef = 0.1  # Weight for reconstruction loss
    args.kl_coef = 0.01  # Weight for latent KL loss

    # Create optimizers
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-8)

    # Create a more gradual learning rate scheduler
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            return 0.9995 ** (epoch - warmup_epochs)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Update the gradient scaling parameters for mixed precision training
    scaler = torch.amp.GradScaler(enabled=str(device) == "cuda", init_scale=2**10)

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
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
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
            agent.load_state_dict(checkpoint)
            print("Using old checkpoint format. Only model weights loaded.")
            
        if args.resume_epoch is not None:
            start_epoch = args.resume_epoch
            print(f"Overriding start epoch to {start_epoch} as specified")
        
        if start_epoch != 151 and args.resume_epoch is None:
            start_epoch = 151
            print("Defaulting to start at epoch 151")
        
        print(f"Resuming training from epoch {start_epoch}")
        print(f"Current best mean reward: {best_mean_reward}")

    print(agent)

    # Create buffer
    buffer = GRPOBuffer(obs_dim, act_dim, args.n_steps, args.n_envs, device, args.gamma, args.gae_lambda, args.latent_dim)

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
            # Store old policy parameters
            old_actor_logstd = agent.actor_logstd.clone().detach()

            # Collect trajectories
            for _ in tqdm(range(0, args.n_steps), desc=f"Epoch {epoch}: Collecting trajectories"):
                global_step_idx += args.n_envs
                obs = next_obs
                terminateds = next_terminateds
                truncateds = next_truncateds

                # Sample actions
                with torch.no_grad():
                    actions, logprobs, _, values, mu_old, reconstructions, latents = agent.get_action_and_value(obs)
                    values = values.reshape(-1)

                # Step environment
                next_obs_np, rewards_np, next_terminateds_np, next_truncateds_np, _ = envs.step(actions.cpu().numpy())

                # Parse to tensors
                next_obs = torch.tensor(np.array(next_obs_np, dtype=np.float32), device=device)
                reward_list.extend(rewards_np)
                rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)
                next_terminateds = torch.as_tensor(next_terminateds_np, dtype=torch.float32, device=device)
                next_truncateds = torch.as_tensor(next_truncateds_np, dtype=torch.float32, device=device)

                # Store step in policy buffer
                buffer.store(obs, actions, rewards, values, terminateds, truncateds, logprobs, mu_old, reconstructions, latents)

            # Calculate advantages and returns
            with torch.no_grad():
                next_values = agent.get_value(next_obs).reshape(1, -1)
                next_terminateds = next_terminateds.reshape(1, -1)
                next_truncateds = next_truncateds.reshape(1, -1)
                traj_adv, traj_ret = buffer.calculate_advantages(next_values, next_terminateds, next_truncateds)

            # Get trajectories
            traj_obs, traj_act, traj_logprob, traj_mu_old, traj_reconst, traj_latent = buffer.get()
            traj_obs = traj_obs.view(-1, *obs_dim)
            traj_act = traj_act.view(-1, *act_dim)
            traj_logprob = traj_logprob.view(-1)
            traj_adv = traj_adv.view(-1)
            traj_ret = traj_ret.view(-1)
            traj_mu_old = traj_mu_old.view(-1, *act_dim)
            traj_reconst = traj_reconst.view(-1, *obs_dim)
            traj_latent = traj_latent.view(-1, args.latent_dim)

            # Normalize advantages
            traj_adv = (traj_adv - traj_adv.mean()) / (traj_adv.std() + 1e-8)

            # Training
            dataset_size = args.n_steps * args.n_envs
            traj_indices = np.arange(dataset_size)
            sum_loss_policy = 0.0
            sum_loss_value = 0.0
            sum_entropy = 0.0
            sum_loss_total = 0.0
            sum_loss_reconst = 0.0
            sum_loss_kl = 0.0
            sum_sinkhorn_div = 0.0
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
                    batch_mu_old = traj_mu_old[batch_indices]
                    batch_reconst = traj_reconst[batch_indices]
                    batch_latent = traj_latent[batch_indices]

                    loss, policy_loss, value_loss, entropy, reconst_loss, kl_loss, sinkhorn_div = grpo_update(
                        agent, optimizer, scaler, batch_obs, batch_actions, batch_returns,
                        batch_old_log_probs, batch_adv, batch_mu_old, batch_reconst, batch_latent,
                        old_actor_logstd, args.clip_ratio, args.vf_coef, args.ent_coef,
                        args.reconst_coef, args.kl_coef, args.beta, args.alpha, n_groups, group_weight
                    )

                    sum_loss_policy += policy_loss
                    sum_loss_value += value_loss
                    sum_entropy += entropy
                    sum_loss_total += loss
                    sum_loss_reconst += reconst_loss
                    sum_loss_kl += kl_loss
                    sum_sinkhorn_div += sinkhorn_div

            # Log losses
            total_loss = sum_loss_total / args.train_iters / (dataset_size / args.batch_size)
            policy_loss = sum_loss_policy / args.train_iters / (dataset_size / args.batch_size)
            value_loss = sum_loss_value / args.train_iters / (dataset_size / args.batch_size)
            entropy = sum_entropy / args.train_iters / (dataset_size / args.batch_size)
            reconst_loss = sum_loss_reconst / args.train_iters / (dataset_size / args.batch_size)
            kl_loss = sum_loss_kl / args.train_iters / (dataset_size / args.batch_size)
            sinkhorn_div = sum_sinkhorn_div / args.train_iters / (dataset_size / args.batch_size)
            writer.add_scalar("loss/total", total_loss, epoch)
            writer.add_scalar("loss/policy", policy_loss, epoch)
            writer.add_scalar("loss/value", value_loss, epoch)
            writer.add_scalar("loss/entropy", entropy, epoch)
            writer.add_scalar("loss/reconstruction", reconst_loss, epoch)
            writer.add_scalar("loss/kl", kl_loss, epoch)
            writer.add_scalar("loss/sinkhorn_div", sinkhorn_div, epoch)

            # Log learning rate
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch)

            # Log rewards
            mean_reward = float(np.mean(reward_list) / args.reward_scale)
            writer.add_scalar("reward/mean", mean_reward, epoch)
            reward_list = []
            print(f"Epoch {epoch} done in {time.time() - start_time:.2f}s, mean reward: {mean_reward:.2f}, "
                  f"total loss: {total_loss:.4f}, policy loss: {policy_loss:.4f}, value loss: {value_loss:.4f}, "
                  f"entropy: {entropy:.4f}, reconst loss: {reconst_loss:.4f}, kl loss: {kl_loss:.4f}, "
                  f"sinkhorn div: {sinkhorn_div:.4f}, learning rate: {scheduler.get_last_lr()[0]:.2e}")
            start_time = time.time()

            # Save model if better
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
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
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Save last model
            torch.save({
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch,
                'global_step': global_step_idx,
                'best_mean_reward': best_mean_reward
            }, os.path.join(checkpoint_dir, "last.pt"))

            # Log video
            if epoch % args.render_epoch == 0:
                log_video(test_env, agent, device, os.path.join(videos_dir, f"epoch_{epoch}.mp4"))

            # Update learning rate
            scheduler.step()

    except Exception as e:
        print(f"An error occurred during training: {e}")
        torch.save({
            'model_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'epoch': epoch,
            'global_step': global_step_idx,
            'best_mean_reward': best_mean_reward
        }, os.path.join(checkpoint_dir, f'error_checkpoint_epoch_{epoch}.pth'))
        raise

    finally:
        envs.close()
        test_env.close()
        writer.close()