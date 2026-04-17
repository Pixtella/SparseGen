import torch

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
)

def get_cameras_from_data_dict(cfg, data, device, slicing_idxs = None):
    # slicing_idxs are the cameras that are kept in the batch
    if slicing_idxs is None:
        slicing_idxs = torch.arange(data["target_Rs"].shape[1])
    example_cameras = FoVPerspectiveCameras(
                R = data["target_Rs"][:, slicing_idxs].reshape([-1, 3, 3]), 
                T = data["target_Ts"][:, slicing_idxs].reshape([-1, 3]),  
                znear = cfg.render.znear,
                zfar = cfg.render.zfar,
                aspect_ratio = cfg.render.aspect_ratio,
                fov = cfg.render.fov,
                device = device,
            )
    return example_cameras
