import torch
import torch.nn as nn
import gsplat
from typing import Dict, Tuple
from pytorch3d.renderer import FoVPerspectiveCameras

class GaussianRenderer(nn.Module):
    """
    Gaussian Splatting Renderer using gsplat library.
    
    This renderer takes 3D Gaussian parameters and camera parameters to produce rendered images
    from multiple viewpoints using FoVPerspectiveCameras from PyTorch3D.
    
    The renderer handles batched Gaussian parameters with shared cameras across the batch.
    For each batch item and camera:
    1. Gets the Gaussian parameters for the current batch
    2. Uses the camera parameters (shared across batch)
    3. Renders using gsplat's rasterize function
    4. Composites with background color
    
    Features:
    - Multi-view rendering with shared cameras
    - Batch processing of Gaussian parameters
    - Configurable background color (white/black)
    - World space coordinate scaling
    - Proper camera matrix handling
    - Input validation
    
    Args:
        cfg: Configuration object containing:
            - data.background: Background color ("white" or "black")
            - data.input_size: (H, W) output image dimensions
            - render.fov: Field of view (for FoVPerspectiveCameras)
            
    Input Shapes:
        - cameras: Contains num_cams cameras shared across batch
        - gaussians: Dictionary of (B, N, *) tensors where:
            B = batch size
            N = number of Gaussians per batch
    
    Output Shape:
        (B, num_cams, H, W, 4) RGBA rendered images
    """
    def __init__(self, cfg, device="cuda"):
        super().__init__()
        self.cfg = cfg
        
        # Set background color based on config
        if self.cfg.data.background == "white":
            self.register_buffer('background_color', torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device))
        elif self.cfg.data.background == "black":
            self.register_buffer('background_color', torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device))
        else:
            raise ValueError(f"Unsupported background color: {self.cfg.data.background}")
        
        self.register_buffer('position_range', torch.tensor(cfg.model.position_range, dtype=torch.float32, device=device))
        self._pyt2gs_trans = torch.diag(torch.tensor([-1, -1, 1, 1], dtype=torch.float32, device=device))
        self.device = device

    def get_camera_matrices(self, cameras) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert PyTorch3D cameras to view and projection matrices required by gsplat.
        
        Args:
            cameras: PyTorch3D FoVPerspectiveCameras object
                    Contains num_cams cameras that are shared across batch
            
        Returns:
            Tuple of:
                viewmats: (num_cams, 4, 4) view matrices
                Ks: (num_cams, 3, 3) camera intrinsic matrices
                
        Note:
            The returned matrices are shared across all batches, as each batch
            is rendered from the same camera viewpoints.
        """
        num_cams = cameras.R.shape[0]
        device = cameras.R.device
        
        # Create view matrices
        viewmats = torch.eye(4, device=device).repeat(num_cams, 1, 1)
        viewmats[:, :3, :3] = cameras.R
        viewmats = viewmats.permute(0, 2, 1)  # (num_cams, 4, 4)
        viewmats[:, :3, 3] = cameras.T
        viewmats = self._pyt2gs_trans[None, :, :] @ viewmats
        viewmats = viewmats.float()
        
        # Create camera intrinsic matrices
        H, W = self.cfg.data.input_size
        fov = self.cfg.render.fov
        focal = W / (2 * torch.tan(torch.tensor(fov * torch.pi / 180.0) / 2))
        Ks = torch.tensor([[focal, 0, W/2],
                          [0, focal, H/2],
                          [0, 0, 1]], device=device, dtype=torch.float32)
        Ks = Ks.unsqueeze(0).repeat(num_cams, 1, 1)

        return viewmats, Ks

    def render_viewpoint(self, gaussians: Dict[str, torch.Tensor], viewmat: torch.Tensor, 
                        K: torch.Tensor, batch_idx: int, xyz_scaled: bool = False) -> torch.Tensor:
        """
        Render Gaussians from a single viewpoint
        Args:
            gaussians: Dictionary of Gaussian parameters
                - xyz: (B, N, 3) positions, either in:
                    - [0,1] range if xyz_scaled=True
                    - world space if xyz_scaled=False
                - opacity: (B, N, 1) opacity values in [0,1]
                - rotation: (B, N, 4) normalized rotation quaternions
                - scale: (B, N, 3) positive scale parameters
                - color: (B, N, 3) RGB colors in [0,1]
            viewmat: (1, 4, 4) view matrix for this viewpoint
            K: (1, 3, 3) camera intrinsic matrix
            batch_idx: Index in the batch to use
            xyz_scaled: If True, positions are in [0,1] and will be scaled to world space.
                       If False, positions are already in world space.
        Returns:
            rendered: (H, W, 4) RGBA rendered image
        """
        H, W = self.cfg.data.input_size
        
        # Get Gaussian parameters for this batch
        means = gaussians['xyz'][batch_idx]  # (N, 3)
        means = means.to(self.device)
                
        # Scale positions if needed
        if xyz_scaled:
            means = means * (self.position_range[None, 3:6] - self.position_range[None, 0:3]) + self.position_range[None, 0:3]        
        # Get other parameters
        quats = gaussians['rotation'][batch_idx].to(self.device)  # (N, 4)
        scales = gaussians['scale'][batch_idx].to(self.device)    # (N, 3)
        colors = gaussians['color'][batch_idx].to(self.device)    # (N, 3)
        opacities = gaussians['opacity'][batch_idx].squeeze(-1).to(self.device)  # (N,)
        
        # Render using gsplat
        # [1, H, W, 3], [1, H, W, 1], info

        colors_out, alphas_out, info = gsplat.rasterization(
            means,      # (N, 3)
            quats,      # (N, 4)
            scales,     # (N, 3)
            opacities,  # (N,)
            colors,     # (N, 3)
            viewmat,    # (C, 4, 4)
            K,          # (C, 3, 3)
            W, H
        )
        # Combine colors and alphas
        rendered = torch.cat([colors_out, alphas_out], dim=-1)
        
        # Deprecated: Composite with background color, done outside
        # rendered[..., :3] = rendered[..., :3] + self.background_color * (1 - rendered[..., 3:])
        
        return rendered

    def forward(self, cameras, gaussians: Dict[str, torch.Tensor], xyz_scaled: bool = False) -> torch.Tensor:
        """
        Render Gaussians from multiple viewpoints
        Args:
            cameras: Camera parameters (FoVPerspectiveCameras)
                Contains num_cams cameras that are shared across the batch
            gaussians: Dictionary containing Gaussian parameters
                - xyz: (B, N, 3) positions, either in:
                    - [0,1] range if xyz_scaled=True (will be scaled to world space)
                    - world space if xyz_scaled=False (used as-is)
                - opacity: (B, N, 1) opacity values in [0,1]
                - rotation: (B, N, 4) normalized rotation quaternions
                - scale: (B, N, 3) positive scale parameters
                - color: (B, N, 3) RGB colors in [0,1]
                where B is batch size and N is number of Gaussians
            xyz_scaled: If True, positions are in [0,1] and will be scaled to world space.
                       If False, positions are already in world space.
        Returns:
            rendered: (B, num_cams, H, W, 4) RGBA rendered images
        """
        # Validate inputs
        if not isinstance(cameras, FoVPerspectiveCameras):
            raise ValueError("cameras must be FoVPerspectiveCameras")
        
        batch_size = gaussians['xyz'].shape[0]
        num_cams = cameras.R.shape[0] // batch_size  # Number of cameras (shared across batch)
        
        # Get camera matrices
        viewmats, Ks = self.get_camera_matrices(cameras)  # (B*N, 4, 4), (B*N, 3, 3)
        viewmats = viewmats.view(batch_size, num_cams, 4, 4).to(self.device)  # (B, num_cams, 4, 4)
        Ks = Ks.view(batch_size, num_cams, 3, 3).to(self.device)            # (B, num_cams, 3, 3)
        
        # Initialize output tensor
        output = []
        
        # Process each batch
        for b in range(batch_size):
            # gsplat does not support batch rendering, so we need to render each batch separately
            # but it supports multiple cameras, so 
            # the input viewmat and K are (num_cams, 4, 4) and (num_cams, 3, 3)

            # Render from all camera viewpoints in once
            rendered = self.render_viewpoint(
                gaussians,
                viewmats[b],  # (num_cams, 4, 4)
                Ks[b],        # (num_cams, 3, 3)
                b,
                xyz_scaled
            )    # (num_cams, H, W, 4)
            output.append(rendered)
        
        # Stack batch dimension
        output = torch.stack(output)  # (B, num_cams, H, W, 4)
        
        return output
