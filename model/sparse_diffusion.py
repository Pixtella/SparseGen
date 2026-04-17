from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from einops import rearrange
from omegaconf import DictConfig
from diffusers import FlowMatchEulerDiscreteScheduler

from .spgen_model import SPGenModel
from .loss_utils import PerceptualLoss
from pytorch_msssim import ssim

class CompositeLoss:
    def __init__(self, cfg: DictConfig):
        """
        Initialize the loss class with configuration.

        Args:
            cfg: Configuration dictionary containing loss weights and options.
        """
        self.cfg = cfg
        self.loss_fn = F.mse_loss if cfg.optimization.loss == "l2" else F.l1_loss
        
        # Get optimization config with defaults
        opt_cfg = cfg.optimization
        self.occ_loss_weight = getattr(opt_cfg, 'occ_loss_weight', 1.0)
        self.perceptual_loss_weight = getattr(opt_cfg, 'perceptual_loss_weight', 0.0)
        self.offset_reg_w = getattr(opt_cfg, 'offset_reg_w', 0.0)
        self.offset_reg_r = getattr(opt_cfg, 'offset_reg_r', 0.0)
        self.ssim_loss_weight = getattr(opt_cfg, 'ssim_loss_weight', 0.0)
        
        # Initialize perceptual loss if enabled
        if self.perceptual_loss_weight > 0.0:
            self.perceptual_loss_module = self._init_frozen_module(PerceptualLoss(device='cuda'))
        else:
            self.perceptual_loss_module = None

    def _init_frozen_module(self, module):
        """Helper method to initialize and freeze a module's parameters."""
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
        return module

    def compute(
        self,
        render_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        render_occ: Optional[torch.Tensor] = None,
        target_occ: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the losses based on the rendered and target images.

        Args:
            render_rgb: Rendered RGB images [b, n_render, 3, h, w].
            target_rgb: Target RGB images [b, n_render, 3, h, w].
            render_occ: Rendered occupancy maps [b, n_render, 1, h, w] (optional).
            target_occ: Target occupancy maps [b, n_render, 1, h, w] (optional).

        Returns:
            Dictionary of computed losses.
        """
        losses = {}

        # Compute RGB loss
        losses["rgb_loss"] = self.loss_fn(render_rgb, target_rgb)

        # Compute perceptual loss if enabled
        if self.perceptual_loss_module is not None:
            # Reshape for perceptual loss computation
            b, n_render, c, h, w = render_rgb.shape
            render_flat = render_rgb.reshape(b * n_render, c, h, w)
            target_flat = target_rgb.reshape(b * n_render, c, h, w)
            
            # Convert from [-1, 1] range to [0, 1] range for perceptual loss
            render_flat = (render_flat + 1.0) * 0.5
            target_flat = (target_flat + 1.0) * 0.5
            
            # Ensure images are in [0, 1] range (clamp any potential numerical issues)
            render_flat = torch.clamp(render_flat, 0.0, 1.0)
            target_flat = torch.clamp(target_flat, 0.0, 1.0)
            
            perceptual_loss = self.perceptual_loss_module(render_flat, target_flat)
            losses["perceptual_loss"] = perceptual_loss * self.perceptual_loss_weight
            if self.ssim_loss_weight > 0.0:
                ssim_loss = 1.0 - ssim(render_flat, target_flat, data_range=1.0, size_average=True)
                losses["perceptual_loss"] += ssim_loss * self.ssim_loss_weight
        else:
            losses["perceptual_loss"] = torch.tensor(0.0, device=render_rgb.device)

        # Compute occupancy loss if applicable
        if render_occ is not None and target_occ is not None:
            losses["occ_loss"] = (
                self.loss_fn(render_occ, target_occ) * self.occ_loss_weight
            )
        else:
            losses["occ_loss"] = torch.tensor(0.0, device=render_rgb.device)
        
        
        # Compute offset regularization loss, if applicable
        if self.offset_reg_w > 0.0 and 'xyz_offset' in kwargs:
            xyz_offset = kwargs['xyz_offset']
            penalty_mask = torch.abs(xyz_offset) > self.offset_reg_r
            penalty = torch.where(penalty_mask, (torch.abs(
                xyz_offset) - self.offset_reg_r) ** 2, torch.zeros_like(xyz_offset))
            losses["offset_reg_loss"] = penalty.mean() * self.offset_reg_w
        else:
            losses["offset_reg_loss"] = torch.tensor(0.0, device=render_rgb.device)

        return losses

class Trainer(nn.Module):
    """
    Diffusion reconstructor that renders a **set** of novel views conditioned on a few
    input views (can optionally drop some views during training).
    """

    def __init__(
        self,
        cfg: DictConfig,
        reconstructor: Union[nn.Module, SPGenModel],
        noise_scheduler: FlowMatchEulerDiscreteScheduler,
        image_size: Union[int, Tuple[int, int]],  # image size (square)
        *,
        loss_type: str = "l2",
        **kwargs,
    ):
        super().__init__()
        self.cfg = cfg
        self.reconstructor = reconstructor
        self.noise_scheduler = noise_scheduler
        self.num_train_timesteps = noise_scheduler.config.num_train_timesteps
        self.image_size = image_size
        self.loss_type = loss_type
        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss
        
        # Instantiate CompositeLoss object
        self.loss_computer = CompositeLoss(cfg)

    def _setup_self_conditioning(self, x, t, cond, idxs_to_keep, kwargs):
        """Setup self-conditioning if enabled in configuration."""
        if not self.cfg.model.unet.self_condition:
            return cond

        cond["x_self_condition"] = torch.zeros_like(x)

        # Apply self-conditioning with 50% probability
        if torch.rand(1) < 0.5:
            if self.cfg.model.unet.self_condition_images == "seen":
                idxs_to_use_for_self_conditioning = idxs_to_keep
            else:
                raise NotImplementedError(
                    "Only 'seen' self-conditioning images supported"
                )

            # Generate self-conditioning with no gradient
            with torch.no_grad():
                model_out = self.reconstructor(
                    x,
                    t,
                    cond,
                    idxs_to_keep=idxs_to_use_for_self_conditioning,
                    return_intermediate=False,
                    **kwargs,
                )['r_imgs']
            cond["x_self_condition"] = model_out[-1, :, cond["x_cond"].shape[1]:, ...]
        return cond

    def forward(
        self, x_start: torch.Tensor, cond: Dict[str, torch.Tensor], t, *args, **kwargs
    ) -> List[torch.Tensor]:
        """
        Perform forward pass for training with noise injection and loss computation.

        Args:
            x_start: Clean input images [batch, views, channels, height, width]
            t: Timesteps for noise scheduling
            cond: Conditioning information dictionary
            loss_weights: Weights for loss computation
            **kwargs: Additional arguments

        Returns:
            List of losses [unseen_loss, seen_loss]
        """
        b, n_views, c, h, w = x_start.shape

        # TODO: to handle non-square images
        assert (
            h == w == self.image_size
        ), "Images must be square and match `image_size`."

        x_start = self._normalize(x_start)  # Normalize to [-1, 1]

        # Step 1: Add noise to images
        noisy_x = self._add_noise_to_images(x_start, t)

        # Step 2: Apply view dropout strategy
        idxs_to_keep, idxs_to_render = self.view_dropout(n_views, cond["x_cond"].shape[1])

        # Step 3: Setup conditioning and optional components
        use_occ = "training_occs" in cond
        cond = self._setup_self_conditioning(noisy_x, t, cond, idxs_to_keep, kwargs)

        # Step 4: Forward pass through reconstructor
        use_intermediate_loss = (
            self.cfg.optimization.intermediate_loss_weight > 0.0
            and self.cfg.model.return_intermediate
        )
        
        ret_dict = self.reconstructor(
            noisy_x,
            t,
            cond,
            idxs_to_keep=idxs_to_keep,
            idxs_to_render=idxs_to_render,
            return_sil=True,
            **kwargs,
        )
        
        model_out, r_sil_out = ret_dict['r_imgs'], ret_dict['r_sils']
        loss_args = {}
        if 'xyz_offset' in ret_dict:
            loss_args['xyz_offset'] = ret_dict['xyz_offset']

        # Step 5: Compute losses
        num_layers = len(model_out)
        render_images = rearrange(model_out, "l b v c h w -> (l b) v c h w" )
        target_images = cond["training_imgs"][:, idxs_to_render, :3] # [b, v, 3, h, w]
        target_images = target_images.repeat(num_layers, 1, 1, 1, 1)  # Repeat for each layer
        if use_occ:
            r_sil_out = rearrange(r_sil_out, "l b v c h w -> (l b) v c h w")
            target_occ = cond["training_occs"][:, idxs_to_render, :1]  # [b, v, 1, h, w]
            target_occ = target_occ.repeat(num_layers, 1, 1, 1, 1)
        return self.compute_losses(render_images, target_images,
                                   r_sil_out,
                                   target_occ if use_occ else None,
                                   **loss_args)

    def compute_losses(
        self,
        render_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        render_occ: Optional[torch.Tensor] = None,  
        target_occ: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute losses based on model output and target images using SparseDiffusionLoss.

        Args:   
            render_rgb: Rendered RGB images [b, n_render, 3, h, w]
            target_rgb: Target RGB images [b, n_render, 3, h, w]
            render_occ: Rendered occupancy maps [b, n_render, 1, h, w] (optional)
            target_occ: Target occupancy maps [b, n_render, 1, h, w] (optional)
        Returns:
            List of computed losses [total_loss, individual_losses_dict]
        """
        # Use the CompositeLoss object to compute losses
        loss_dict = self.loss_computer.compute(
            render_rgb=render_rgb,
            target_rgb=target_rgb,
            render_occ=render_occ,
            target_occ=target_occ,
            **kwargs
        )
        
        return [v for k, v in loss_dict.items() if v is not None]

    @torch.no_grad()
    def log_visualisations(self, cond: Dict[str, Any], iteration: int, split: str, noise_scheduler):

        x_start = cond["x_in"] * 2 - 1
        
        # for every pass through the model we want to show the inputs, outputs
        # and a red border if the image was dropped out
        vis_columns = []

        # sample noise for the noisy images
        b, Cond, c, h, w = x_start.shape

        t = torch.randint(0, self.num_train_timesteps, (b,), device=x_start.device) 
        indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,)).long()
        t = noise_scheduler.timesteps[indices].to(x_start.device)

        x = self._add_noise_to_images(x_start, t)   
        
        # view dropout
        idxs_to_keep, idxs_to_render = self.view_dropout(Cond, num_clean=cond["x_cond"].shape[1])

        rows = []
        imgs_in = torch.cat([cond["x_cond"][0], x[0]], 
                            dim = 0)[:, :3, ...].permute(0, 2, 3, 1).cpu() * 0.5 + 0.5
        for img_idx in range(cond["x_cond"].shape[1] + Cond):
            img_in = imgs_in[img_idx].clone()
            if img_idx in idxs_to_keep:
                img_in[:5, :5, :] = 1.0
            else:
                img_in[:5, :5, :] = 0.0
            rows.append(img_in) 
        vis_columns.append(torch.vstack(rows))

        # add occ rows if present
        if "training_occs" in cond:
            rows = []
            for occ_idx in range(cond["x_cond"].shape[1] + Cond):
                occ_in = cond["training_occs"][0, occ_idx].cpu().permute(1, 2, 0).expand(h, w, 3)
                rows.append(occ_in)
            vis_columns.append(torch.vstack(rows))
 
        # optionally get self-conditioning
        if self.cfg.model.unet.self_condition:
            cond["x_self_condition"] = torch.zeros_like(x)
            # this forward pass has to render all images in the set
            ret_dict = self.reconstructor(x, t, cond,
                                            idxs_to_keep=idxs_to_keep,
                                            return_sil=True,
                                            return_intermediate=False
                                            )
            volume_model_out, r_sil_out = ret_dict['r_imgs'], ret_dict['r_sils']
            volume_model_out = volume_model_out[-1] # [B, N, 3, H, W]
            r_sil_out = r_sil_out[-1].squeeze(2) # [B, N, H, W]
            # visualise output of self-conditioning pass: renders and silhouettes
            rows = []
            for img_idx in range(volume_model_out.shape[1]):
                rows.append(volume_model_out[0, img_idx, :3, ...].permute(1, 2, 0).cpu()* 0.5 + 0.5) 
            vis_columns.append(torch.vstack(rows))
            rows = []
            r_sil_out = r_sil_out.reshape(b, Cond+cond["x_cond"].shape[1], h, w, 1)
            for img_idx in range(r_sil_out.shape[1]):
                rows.append(r_sil_out[0, img_idx, ...].expand(h, w, 3).cpu()) 
            vis_columns.append(torch.vstack(rows))

            cond["x_self_condition"] = volume_model_out[:, cond["x_cond"].shape[1]:,
                                                        ...]
        
        # model forward pass
        ret_dict = self.reconstructor(x, t, cond,
                                        idxs_to_keep=idxs_to_keep,
                                        return_sil=True,
                                        return_intermediate=False)
        volume_model_out, r_sil_out = ret_dict['r_imgs'], ret_dict['r_sils']
        volume_model_out = volume_model_out[-1] # [B, N, 3, H, W]
        r_sil_out = r_sil_out[-1].squeeze(2) # [B, N, H, W]
        rows = []
        was_dropped_out = torch.logical_not(torch.isin(torch.arange(imgs_in.shape[0]), 
                                                    idxs_to_keep))
        for img_idx in range(volume_model_out.shape[1]):
            img_out = volume_model_out[0, img_idx, :3, ...].permute(1, 2, 0).cpu()* 0.5 + 0.5
            if was_dropped_out[img_idx]:
                img_out[:5, :5, :] = 0.0
            else:
                img_out[:5, :5, :] = 1.0
            rows.append(img_out)
        vis_columns.append(torch.vstack(rows))
        rows = []
        r_sil_out = r_sil_out.reshape(b, Cond+cond["x_cond"].shape[1], h, w, 1)
        for img_idx in range(r_sil_out.shape[1]):
            rows.append(r_sil_out[0, img_idx, ...].expand(h, w, 3).cpu()) 
        vis_columns.append(torch.vstack(rows))
        
        # log the visualisations
        im_log = wandb.Image(np.clip(torch.hstack(vis_columns).numpy(), 0.0, 1.0), caption="Vis")
        wandb.log({"{}_vis".format(split): im_log}, step=iteration)
            
    # ...........................................................................
    # Core diffusion logic (loss, prediction, sampling)
    # ...........................................................................
    # ---- helpers --------------------------------------------------------------
    def _add_noise_to_images(self, x_start, t):
        """Add noise to input images using the noise scheduler.

        Args:
            x_start: Clean input images [batch, views, channels, height, width]
            t: Timesteps for noise scheduling
        Returns:
            Noisy images with added noise [batch, views, channels, height, width]
        """

        def get_sigmas(
            noise_scheduler, timesteps, n_dim=4, dtype=torch.float32, device="cuda"
        ):
            sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
            schedule_timesteps = noise_scheduler.timesteps.to(device=device)
            timesteps = timesteps.to(device=device)
            try:
                step_indices = [
                    (schedule_timesteps == t).nonzero().item() for t in timesteps
                ]
            except:
                import pdb

                pdb.set_trace()
            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma

        batch_size, num_views, c, h, w = x_start.shape
        noise = torch.randn_like(x_start)

        # Flatten for noise scheduling
        x_start_flat = x_start.reshape(batch_size * num_views, *x_start.shape[2:])
        t_flat = (
            t.unsqueeze(1).expand(batch_size, num_views).reshape(batch_size * num_views)
        )
        noise_flat = noise.reshape(batch_size * num_views, *noise.shape[2:])
        
        # Add noise using appropriate method
        if hasattr(self.noise_scheduler, "add_noise"):
            x_flat = self.noise_scheduler.add_noise(x_start_flat, noise_flat, t_flat)
        else:
            sigmas = get_sigmas(
                self.noise_scheduler,
                t_flat,
                n_dim=x_start_flat.ndim,
                dtype=x_start_flat.dtype,
                device=x_start_flat.device,
            )
            x_flat = (1 - sigmas) * x_start_flat + sigmas * noise_flat

        return x_flat.reshape(x_start.shape)

    def view_dropout(self, num_views: int, num_clean: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the indices of images that should be used for reconstruction
        and the indices of images that should be rendered.
        """
        clean_idx = torch.arange(num_views + num_clean)[:num_clean]
        noisy_idxs = torch.arange(num_views + num_clean)[num_clean:]

        idxs_to_keep = None
        idxs_to_render = None

        clean_dropout_rand = torch.rand(1) - self.cfg.optimization.hard_mining_proportion + 1e-6
        
        if clean_dropout_rand > 0:
            # keep some of the clean views
            num_keep_clean = clean_dropout_rand / (1 - self.cfg.optimization.hard_mining_proportion)
            num_keep_clean = int(np.ceil(num_keep_clean * num_clean))
            idxs_to_keep = clean_idx[:num_keep_clean]
            clean_idx_dropped_out = clean_idx[num_keep_clean:]
        else:
            # drop all clean views
            clean_idx_dropped_out = clean_idx
        
        if self.cfg.optimization.noisy_dropout_proportion == 0.5:
            rand_start = torch.randint(num_views, (1,))
            idxs_to_keep = torch.cat([idxs_to_keep, noisy_idxs[rand_start:]]) if idxs_to_keep is not None else noisy_idxs[rand_start:]
        elif self.cfg.optimization.noisy_dropout_proportion == 1.0:
            idxs_to_keep = idxs_to_keep
        else:
            raise NotImplementedError("Only 0.5 noisy dropout proportion supported")
        
        if self.cfg.optimization.penalize == "seen":
            idxs_to_render = idxs_to_keep
        elif self.cfg.optimization.penalize == "all":
            idxs_to_render = torch.arange(num_views + num_clean)
        elif self.cfg.optimization.penalize == "seen_and_one_more":
            if clean_dropout_rand <= 0:
                idxs_to_render = torch.cat([idxs_to_keep, clean_idx_dropped_out[0:1]])
            else:
                idxs_to_render = torch.cat([idxs_to_keep, noisy_idxs[1:]])
        else:
            raise NotImplementedError


        return idxs_to_keep, idxs_to_render
            

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 - 1.0  # map [0,1] → [-1,1]
