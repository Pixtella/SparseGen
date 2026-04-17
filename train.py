# ----------------------------------------------------------------------
#  Copyright (c) 2025
# ----------------------------------------------------------------------

from pathlib import Path
from itertools import cycle
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from diffusers import FlowMatchEulerDiscreteScheduler
from data_manager import get_data_manager
from model.spgen_model import SPGenModel
from model.sparse_diffusion import Trainer
from ema_pytorch import EMA
from utils import set_seed
import torch.distributed as dist

from utils import setup_ddp, is_main_process, barrier
import math

# ----------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------

def create_reconstructor(cfg: DictConfig):
    """Return the query reconstructor implementation."""
    if cfg.model.type != "query":
        raise ValueError(f"Only model.type='query' is supported, got {cfg.model.type}")
    return SPGenModel(cfg)

def build_dataloaders(cfg: DictConfig):
    """Build train / val dataloaders with DistributedSamplers."""
    train_ds = get_data_manager(cfg, split="train")
    val_single = cfg.data.dataset_type == "srn"
    val_ds = get_data_manager(
        cfg, split="val",
        convert_to_single_conditioning=val_single,
        convert_to_double_conditioning=False,
        for_training=True,
    )

    # Distributed samplers
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
        val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler   = None

    loader_args = dict(
        batch_size=cfg.optimization.batch_size,
        num_workers=cfg.data.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=cfg.data.get("num_workers", 4) > 0,
    )

    train_dataloader = DataLoader(train_ds, sampler=train_sampler, **loader_args)
    val_dataloader   = DataLoader(val_ds,   sampler=val_sampler,   **loader_args)

    return train_dataloader, val_dataloader

def create_parameter_groups(model: torch.nn.Module, cfg: DictConfig):
    """Create parameter groups with different learning rates for different model components."""
    base_lr = cfg.optimization.lr
    lr_multipliers = cfg.optimization.get("lr_multipliers", {})
    
    # Default multipliers if not specified
    feature_extractor_mult = lr_multipliers.get("feature_extractor", 1.0)
    
    parameter_groups = []
    
    # Get all trainable parameters with their names
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    
    # Group parameters by component
    feature_extractor_params = []
    other_params = []
    
    for name, param in named_params:
        if "feature_extractor" in name:
            feature_extractor_params.append(param)
        else:
            other_params.append(param)
    
    # Create parameter groups with different learning rates
    if feature_extractor_params:
        parameter_groups.append({
            'params': feature_extractor_params,
            'lr': base_lr * feature_extractor_mult,
        })
    if other_params:
        parameter_groups.append({
            'params': other_params,
            'lr': base_lr,
        })

    return parameter_groups

def log_to_wandb(loss_dict: Dict[str, float], iteration: int):
    """Small wrapper to avoid repetitive wandb.log calls."""
    wandb.log(loss_dict, step=iteration)

# ----------------------------------------------------------------------
#  Main training routine
# ----------------------------------------------------------------------

def train(cfg: DictConfig):
    # ------------------------------------------------------------------
    # DDP / device setup
    # ------------------------------------------------------------------
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True  # same as "high" matmul precision
    torch.backends.cudnn.allow_tf32 = True  # same as "high" convolution precision
    set_seed(cfg.general.random_seed)

    device, local_rank = setup_ddp()
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    # ---------------- W&B (rank-0 only) ----------------
    if is_main_process():
        run_dir = hydra.core.hydra_config.HydraConfig.get().run.dir
        wandb.init(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            group="tr",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_dir[-19:],
        )

    # ---------------- Models ----------------
    reconstructor = create_reconstructor(cfg).to(device)

    if cfg.model.diffuser.type != "RF":
        raise NotImplementedError(f"Diffuser type {cfg.model.diffuser.type} is not implemented.")
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=cfg.model.diffuser.steps,
        shift=1.0,
        use_dynamic_shifting=False,
    )

    diffuser = Trainer(
        cfg,
        reconstructor,
        noise_scheduler=noise_scheduler,
        image_size=cfg.data.input_size[0],
        loss_type=cfg.optimization.loss,
    ).to(device)

    # ---------------- Optimizer & EMA ----------------
    # Create parameter groups with different learning rates
    parameter_groups = create_parameter_groups(diffuser, cfg)
    
    if is_main_process():
        print("Parameter groups:")
        for i, group in enumerate(parameter_groups):
            n_params = sum(p.numel() for p in group['params'])
            print(f"  {group.get('name', f'group_{i}')}: {n_params:,} params, lr={group['lr']:.6f}")
    
    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=tuple(cfg.optimization.betas),
        fused=False,
    )

    # EMA on rank-0 only; we update with the DDP-wrapped model later
    ema = None
    if cfg.optimization.ema.use and is_main_process():
        ema = EMA(diffuser, beta=cfg.optimization.ema.decay,
                  update_every=cfg.optimization.ema.update_every)
        ema.to(device)

    # ---------------- DDP wrap ----------------
    diffuser = torch.nn.parallel.DistributedDataParallel(
        diffuser,
        device_ids=[local_rank],
        output_device=local_rank,
        # find_unused_parameters=True,  # keep behavior similar to Accelerate
    )

    # ---------------- Data ----------------
    train_dataloader, val_dataloader = build_dataloaders(cfg)
    val_iter = cycle(val_dataloader)

    if is_main_process():
        print("▶ Training from scratch.")

    iteration = 0

    # ---------------- Loop ----------------
    n_iter = cfg.optimization.n_iter
    progress = tqdm(
        range(iteration, n_iter),
        disable=not is_main_process(),
        bar_format='{l_bar}{bar:40}{r_bar}',
    )
    cur_epoch = 0
    if isinstance(train_dataloader.sampler, DistributedSampler):
        train_dataloader.sampler.set_epoch(cur_epoch)
    train_iter = iter(train_dataloader)
    while iteration < n_iter:
    
        diffuser.train()
        iteration += 1
        # Data loading
        try:
            batch = next(train_iter)
        except StopIteration:
            cur_epoch += 1
            if isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(cur_epoch)
            train_iter = iter(train_dataloader)  # Reset when epoch ends
            batch = next(train_iter)

        # move tensors to device (minimal example: expects 'x_in' in batch)
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch["x_in"].shape[0],)).long()
        t = noise_scheduler.timesteps[indices].to(device)

        # import pdb; pdb.set_trace()
        # Forward pass
        losses: List[torch.Tensor] = diffuser(batch["x_in"], batch, iteration=iteration, t=t)
        total_loss = torch.stack([l for l in losses if not torch.isnan(l)]).sum()

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # EMA update
        if ema is not None and is_main_process():
            ema.update()

        # -- validation every 10 iterations (cheap)
        if (iteration + 1) % 10 == 0:
            diffuser.eval()
            with torch.no_grad():
                val_batch = next(val_iter)
                val_batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                             for k, v in val_batch.items()}
                indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (val_batch["x_in"].shape[0],)).long()
                t = noise_scheduler.timesteps[indices].to(device)
                val_losses = diffuser(val_batch["x_in"], val_batch, t=t)
                val_loss   = torch.stack([l for l in val_losses if not torch.isnan(l)]).sum()

            if is_main_process():
                # Get current learning rate
                current_lr = optimizer.param_groups[-1]['lr']
                
                log_to_wandb(
                    {
                        "loss_total": math.log(float(total_loss.detach().cpu()) + 1e-5),
                        "val_total":  math.log(float(val_loss.detach().cpu()) + 1e-5),
                        "learning_rate": current_lr,
                        **{f"loss_{i}": math.log(float(l.detach().cpu()) + 1e-5) for i, l in enumerate(losses)},
                    },
                    iteration,
                )
                progress.set_postfix(loss=float(total_loss), vloss=float(val_loss), lr=current_lr)
                progress.update(10)

        # -- heavy visualisation / checkpointing
        # if (iteration % 10 == 0) and is_main_process():
        if (iteration % max(1, n_iter // 100) == 0) and is_main_process():
        # if True:
            # Visualise the current iteration
            diffuser.module.log_visualisations(
                batch, iteration=iteration, split="train", noise_scheduler=noise_scheduler
            )
            val_batch = next(val_iter)
            val_batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                         for k, v in val_batch.items()}
            diffuser.module.log_visualisations(
                val_batch, iteration=iteration, split="val", noise_scheduler=noise_scheduler
            )

        if (iteration + 1) % cfg.optimization.save_every == 0 and is_main_process():
            to_save_model = ema.ema_model if ema else diffuser.module
            ckpt = {
                "diffuser": to_save_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, output_dir / f"model.pth")
            print(f"✔ Saved checkpoint @ iter={iteration}")

    if is_main_process():
        wandb.finish()
    barrier()

# ----------------------------------------------------------------------
#  Entry point
# ----------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="default_config")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()
