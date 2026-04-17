import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any
from . import get_cameras_from_data_dict
from .model_utils import SinePositionEmbedding3D, pos2posemb3d, inverse_sigmoid
from .renderer_gs import GaussianRenderer
import math
from .sp_trans import SPGenTransformer
from .extractors import ViTExtractor

class SPGenModel(nn.Module):
    def __init__(self, cfg):
        super(SPGenModel, self).__init__()
        print('SPGenModel Instantiated')
        self.cfg = cfg
        self.max_scale = cfg.model.max_scale
        self.renderer = GaussianRenderer(cfg)
        self.feature_extractor = ViTExtractor(
            pretrained=self.cfg.model.feature_extractor_2d.pretrained
        )

        # dummy attributes expected by GaussianDiffusion
        self.random_or_learned_sinusoidal_cond = False
        self.channels = 3
        self.self_condition = False
        
        self.znear = cfg.render.znear
        self.zfar = cfg.render.zfar
        self.register_buffer('position_range', torch.tensor(cfg.model.position_range, dtype=torch.float32))
        # self.position_range = cfg.model.position_range # list of 6 floats
        
        self._init_query_network()
        
        self.input_proj = nn.Conv2d(
                self.feature_extractor.output_dim, self.embed_dims, kernel_size=1)
        self._init_gs_decoder()
        self._init_weights()

    def _init_weights(self):
        if self.cfg.model.querty_init == "uniform":
            nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        elif self.cfg.model.querty_init == "trunc_normal":
            nn.init.trunc_normal_(self.reference_points.weight.data, mean=0.5, std=0.1, a=0.0, b=1.0)
        elif self.cfg.model.querty_init == "constant":
            nn.init.constant_(self.reference_points.weight.data[:, 0], 0.5)
        else:
            raise ValueError(f"Unknown query initialization method: {self.cfg.model.querty_init}")
        # transformer weights already initialized in the transformer class
        # init the decoder weights
        for layer in self.gs_decoder:
            for param in layer.parameters():
                if param.data.dim() >= 2:
                    nn.init.xavier_uniform_(param.data)
                elif param.data.dim() == 1:
                    nn.init.normal_(param.data)

    def _init_query_network(self):
        # Query network
        self.embed_dims = self.cfg.model.embed_dims
        self.depth_num = self.cfg.model.depth_num
        self.position_dim = 3 * self.depth_num
        self.num_query = self.cfg.model.num_query
        self.num_gs_per_query = self.cfg.model.num_gs_per_query 
        self.reference_points = nn.Embedding(self.num_query, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        
        self.positional_encoding = self.build_position_encoding(self.embed_dims // 2)
        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.embed_dims*3//2, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.position_encoder = nn.Sequential(
            nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )

        self.transformer = SPGenTransformer(hidden_dim=self.embed_dims,
                                           num_encoder_layers=self.cfg.model.num_encoder_layers,
                                           num_decoder_layers=self.cfg.model.num_decoder_layers,
                                           num_heads=self.cfg.model.num_heads,
                                           return_intermediate=self.cfg.model.return_intermediate,
                                           cfg=self.cfg)

    def _init_gs_decoder(self):
        # Define output dimensions for each parameter type
        xyz_dim = self.num_gs_per_query * 3 # 3D position
        opacity_dim = self.num_gs_per_query * 1 # Transparency
        rotation_dim = self.num_gs_per_query * 4 # Quaternion rotation
        scale_dim = self.num_gs_per_query * 3  # 3D scale
        color_dim = self.num_gs_per_query * 3  # RGB color
        
        self.xyz_activation = nn.Sigmoid() # nn.Tanh() 
        self.opacity_activation = nn.Sigmoid()
        self.rotation_activation = lambda x: F.normalize(x, dim=-1)  # Normalize to unit quaternion
        self.scale_activation = torch.exp  # Ensure positive scale values
        self.color_activation = nn.Sigmoid()  # Bound color to [0,1]
        
        # Create decoder architecture for a single layer
        def create_decoder():
            return nn.ModuleDict({
                'xyz': nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, xyz_dim),
                    # nn.Sigmoid()  # Bound positions to [0,1]
                ),
                'opacity': nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims//2),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims//2, opacity_dim),
                ),
                'rotation': nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, rotation_dim)
                    # Quaternion normalization will be done during forward pass
                ),
                'scale': nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims//2),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims//2, scale_dim),
                    # nn.Softplus()  # Ensure positive scale values
                ),
                'color': nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, color_dim),
                )
            })
        
        if self.cfg.model.return_intermediate:
            # Create decoders for each transformer layer
            self.gs_decoder = nn.ModuleList([
                create_decoder() for _ in range(self.cfg.model.num_decoder_layers)
            ])
        else:
            # Create single decoder for final layer
            self.gs_decoder = nn.ModuleList([create_decoder()])

    @torch.autocast(device_type='cuda', enabled=False)
    def decode_gaussian_parameters(self, outs_dec, camreas, reference_points=None):
        """
        Decode transformer outputs into Gaussian parameters for each layer
        Args:
            outs_dec: Transformer decoder outputs
                If return_intermediate: [num_dec_layers, B, num_query, embed_dims]
                Else: [1, B, num_query, embed_dims]
            cameras: Cameras for each view
                Used for transforming reference points to world space
            reference_points: Optional reference points for Gaussian parameters
                If provided, will be added to the decoded positions
                Shape: [B, num_query, 3]
                
        Returns:
            List of dictionaries containing decoded parameters for each layer:
            [
                {
                    'xyz': Position parameters [B, num_query, 3],
                    'opacity': Opacity parameters [B, num_query, 1],
                    'rotation': Rotation parameters [B, num_query, 4],
                    'scale': Scale parameters [B, num_query, 3]
                },
                ...
            ]
        """
        _, B, num_query, embed_dims = outs_dec.shape
        assert num_query == self.num_query, f"Expected num_query {self.num_query}, but got {num_query}"

        gaussian_params_list = []
        
        # Handle both single and multi-layer cases
        num_layers = len(outs_dec) if self.cfg.model.return_intermediate else 1
        
        for layer_idx in range(num_layers):
            layer_out = outs_dec[layer_idx].float()  # [B, num_query, embed_dims]
            
            # Get decoder for this layer
            decoder = self.gs_decoder[layer_idx]
            
            xyz_decoded = decoder['xyz'](layer_out) # [B, num_query, self.num_gs_per_query*3]
            # Reshape to [B, num_query, num_gs_per_query, 3]
            xyz_decoded = xyz_decoded.view(B, num_query, self.num_gs_per_query, 3)
            
            if reference_points is not None:
                # use xyz_decoded as offsets to reference points
                # reference_points: [B, num_query, 3]
                # xyz_decoded: [B, num_query, num_gs_per_query, 3]
                gs_means = self.xyz_activation(inverse_sigmoid(reference_points)[:, :, None, :] + xyz_decoded)
            else:
                gs_means = self.xyz_activation(xyz_decoded)
            
            gs_means = gs_means.view(B, num_query * self.num_gs_per_query, 3)
            
            # Decode opacity
            gs_opacity = decoder['opacity'](layer_out) # [B, num_query, self.num_gs_per_query*1]
            gs_opacity = self.opacity_activation(gs_opacity - 2.0)
            # Reshape to [B, num_query, num_gs_per_query, 1]
            gs_opacity = gs_opacity.view(B, num_query * self.num_gs_per_query, 1)
            
            # Normalize rotation to unit quaternion
            gs_rotation = decoder['rotation'](layer_out) # [B, num_query, self.num_gs_per_query*4]
            gs_rotation = gs_rotation.view(B, num_query * self.num_gs_per_query, 4)
            gs_rotation = self.rotation_activation(gs_rotation) # [B, num_query * num_gs_per_query, 4]
            
            # Decode scale
            gs_scale = decoder['scale'](layer_out) # [B, num_query, self.num_gs_per_query*3]
            gs_scale = self.scale_activation(gs_scale - 2.3)
            gs_scale = gs_scale.clamp_max(self.max_scale)
            # gs_scale = self.scale_activation(gs_scale) * self.max_scale
            # Reshape to [B, num_query, num_gs_per_query, 3]
            gs_scale = gs_scale.view(B, num_query * self.num_gs_per_query, 3)
            
            gs_color = decoder['color'](layer_out) # [B, num_query, self.num_gs_per_query*3]
            gs_color = self.color_activation(gs_color) # [B, num_query, self.num_gs_per_query*3]
            gs_color = gs_color.view(B, num_query * self.num_gs_per_query, 3)
            
            # Decode parameters
            params = {
                'xyz': gs_means,
                'opacity': gs_opacity,
                'rotation': gs_rotation,
                'scale': gs_scale,
                'color': gs_color,
                'xyz_offset': xyz_decoded,  # [B, num_query, num_gs_per_query, 3]
            }
            
            gaussian_params_list.append(params)
        
        return gaussian_params_list

    def build_position_encoding(self, embed_dims=256):
        return SinePositionEmbedding3D(embed_dims, normalize=True)

    def position_embeding(self, img_feats, cameras, pad_shape = [128, 128], masks=None):
        eps = 1e-5
        pad_h, pad_w = pad_shape
        B, N, C, h, w = img_feats.shape
        coords_h = (torch.arange(h, device=img_feats[0].device).float() + 0.5) / h * pad_h
        coords_w = (torch.arange(w, device=img_feats[0].device).float() + 0.5) / w * pad_w

        index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
        bin_size = (self.zfar - self.znear) / self.depth_num
        coords_d = self.znear + bin_size * index

        d = coords_d.shape[0]
        
        if cameras.in_ndc(): # convert to NDC if needed
            # https://pytorch3d.org/docs/cameras
            # NDC coordinates system in PyTorch3D:
            #             up Y    Z front
            #                ^   ^
            #                |  /
            #                | /
            # left X <------ 0
            coords_h = -2 * (coords_h / pad_h) + 1  # x_ndc (PyTorch3D +X is left) 
            coords_w = -2 * (coords_w / pad_w) + 1  # y_ndc (PyTorch3D +Y is up)    
              
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d], indexing='ij'))
        coords = coords.permute(1, 2, 3, 0).contiguous() # [w, h, d, 3]
        coords = coords.reshape(-1, 3) # [w*h*d, 3]
        
        # NOTE: FOV cameras are in ndc coordinates
        # pytorch3d cameras: https://pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html
        
        if cameras.in_ndc():
            coords = coords.unsqueeze(0).repeat(B*N, 1, 1) # [B*N, w*h*d, 3]
            coords3d = cameras.unproject_points(coords, world_coordinates=True)  # [B*N, w*h*d, 3]
        else:
            # TODO: implement this for non-ndc cameras
            raise NotImplementedError("Only NDC cameras are supported for now")

        coords3d = coords3d.view(B, N, w, h, d, 3) # [B, N, w, h, d, 3]

        coords3d = (coords3d - self.position_range[:3]) / (self.position_range[3:] - self.position_range[:3])
        

        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B*N, -1, h, w) # [B*N, d*3, h, w]
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d) # [B*N, embed_dims, h, w]
        return coords_position_embeding.view(B, N, self.embed_dims, h, w)

    def get_pos_embedding(self, source_images_feats, source_cameras, H, W, masks=None):
        """
        Args:
            source_images_feats: [B, N, C, h, w]
            source_cameras: pytorch3d cameras, length = N
            H: height of the image
            W: width of the image
            masks: [B, N, h, w]
        Returns:
            pos_embed: [B, N, embed_dims, h, w] 
        """
        B, num_cams, C, h, w = source_images_feats.shape
        coords_position_embeding = self.position_embeding(source_images_feats, source_cameras, [H, W], masks)
        pos_embed = coords_position_embeding
        sin_embed = self.positional_encoding(masks)
        sin_embed = sin_embed.view(B, num_cams, -1, h, w)
        sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(source_images_feats.size())
        pos_embed = pos_embed + sin_embed
        return pos_embed # [B, N, embed_dims, h, w]
    
    def forward(self, imgs, timesteps, cond,
                idxs_to_keep=None, idxs_to_render=None,
                return_sil=False,
                return_intermediate: Optional[Any] = None, **kwargs):
        """ Forward pass of the SPGenModel model.    
        Args:
            imgs: Input images, shape [B, N, C, H, W]
            timesteps: Timestep tensor, shape [B, N]
            cond: Dictionary containing conditioning information, including:
                - x_cond: Conditional input images, shape [B, N, C, H, W]
                - pose_embed: Pose embeddings, shape [B, N, embed_dims, H, W]
                - x_self_condition: Self-conditioning images (optional), shape [B, N, C, H, W]
                - target_Rs: Target camera rotations (optional), shape [B, M, 3, 3]
            idxs_to_keep: Indices of images to keep for processing, shape [N_keep]
            idxs_to_render: Indices of images to render, shape [N_render]   
            return_sil: Whether to return the silhouette images (default: False)
            return_intermediate: Whether to return intermediate outputs (default: False)
        Returns:
            r_imgs: Rendered images, shape [num_layers, B, N, 3, H, W], num_layers depends on return_intermediate
            r_sils: Rendered silhouette images, shape [num_layers, B, N, 1, H, W] if return_sil is True
        """
        # ============ Building input images ============
        imgs = torch.cat([cond["x_cond"], imgs], dim=1)
        imgs = torch.cat([imgs, cond["pose_embed"]], dim=2)
        if self.cfg.model.unet.self_condition:
            self_conditioning = torch.cat([cond["x_cond"],
                                            cond["x_self_condition"]],
                                            dim=1)
            imgs = torch.cat([imgs, self_conditioning], dim=2)
        noisy_imgs_in = imgs.shape[1] - cond["x_cond"].shape[1]

        # ============ Building source cameras, images and timesteps
        if idxs_to_keep is None:
            idxs_to_keep = torch.arange(imgs.shape[1])
        source_cameras = get_cameras_from_data_dict(self.cfg, 
                                                    cond, 
                                                    imgs.device, 
                                                    idxs_to_keep)
        source_images = imgs[:, idxs_to_keep, ...]
        B, Cond, C, H, W = imgs.shape

        # ============ Preparing outputs in the target shape ============
        if idxs_to_render is None:
            idxs_to_render = torch.arange(cond["target_Rs"].shape[1])
        target_cameras = get_cameras_from_data_dict(self.cfg, 
                                                    cond, 
                                                    imgs.device,
                                                    idxs_to_render)
        Renders = len(idxs_to_render)
        
        if noisy_imgs_in == 0:
            timesteps = torch.zeros((B, cond["x_cond"].shape[1]), 
                                    device=imgs.device)[:, idxs_to_keep, ...]
        else:
            timesteps = timesteps.unsqueeze(1).expand(B, noisy_imgs_in)
            timesteps = torch.cat([timesteps[:, :1].expand(B, cond["x_cond"].shape[1]) * 0,
                                   timesteps], dim=1)[..., idxs_to_keep]

        # ============ Image feature extraction ============
        source_images_feats = self.feature_extractor(source_images, timesteps) # [B, N, C, H, W]
        B, num_cams , _, h, w = source_images_feats.shape
        source_images_feats = self.input_proj(source_images_feats.flatten(0, 1)).view(B, num_cams, -1, h, w) # [B, N, embed_dims, H, W]
        
        # ============ Query update ============
        masks = source_images_feats.new_ones(B, num_cams, h, w).to(torch.bool) # [B, N, h, w]
        pos_embed = self.get_pos_embedding(source_images_feats, source_cameras, H, W, masks) # [B, N, embed_dims, h, w]
        reference_points = self.reference_points.weight[None, ...].repeat(B, 1, 1)
        query_embeds = self.query_embedding(
            pos2posemb3d(reference_points.view(-1, 3), num_pos_feats=self.embed_dims // 2)
        ).view(B, self.num_query, -1)
        self_attn_mask = None
        
        tf_ret = self.transformer(
            source_images_feats,
            masks,
            query_embeds,
            pos_embed,
            timesteps=timesteps,
            self_attn_mask=self_attn_mask,
        )
        
        outs_dec = tf_ret['output']
        outs_dec = torch.nan_to_num(outs_dec) 
        
        
        # ============ Gaussian decoding ============
        with torch.autocast(device_type="cuda", enabled=False):
            gaussian_params_list = self.decode_gaussian_parameters(outs_dec, source_cameras, reference_points)
        
        # TODO: Use gaussian_params_list for rendering
        # Each element in gaussian_params_list contains parameters for one layer:
        # - xyz: [B, num_query * num_gs_per_query, 3] positions
        # - opacity: [B, num_query * num_gs_per_query, 1] transparency
        # - rotation: [B, num_query * num_gs_per_query, 4] normalized quaternions
        # - scale: [B, num_query * num_gs_per_query, 3] positive scale values
    
        # ============ Rendering ============
        if return_intermediate is None:
            # if not parameter specified, use the value from config
            return_intermediate = self.cfg.model.return_intermediate
        
        if not return_intermediate:
            gaussian_params_list = [gaussian_params_list[-1]]
        r_img_list = []
        r_sil_list = []
        for params in gaussian_params_list:
            r_img, r_sil = self.renderer(
                cameras=target_cameras,
                gaussians=params,
                xyz_scaled=True
            ).split([3, 1], dim=-1)
            
            r_img = r_img + cond["background"][:, idxs_to_render, ...] * (1. - r_sil)
            r_img = r_img.permute(0, 1, 4, 2, 3) # [B, N, H, W, 3] -> [B, N, 3, H, W]
            r_sil = r_sil.permute(0, 1, 4, 2, 3) # [B, N, H, W, 1] -> [B, N, 1, H, W]
            r_img = r_img * 2 - 1 # [0-1] -> [-1,1]
            r_img_list.append(r_img)
            r_sil_list.append(r_sil)
        
        r_images = torch.stack(r_img_list, dim=0) # [num_layers, B, Renders, 3, H, W]
        
        ret_dict = {
            'r_imgs': r_images,
            'xyz_offset': torch.stack([params['xyz_offset'] for params in gaussian_params_list], dim=0),
            'xyz': gaussian_params_list[-1]['xyz'],
            'ref_pts': reference_points,
        }
        if return_sil:
            ret_dict['r_sils'] = torch.stack(r_sil_list, dim=0)  # [num_layers, B, Renders, 1, H, W]
        return ret_dict
