import torch
import torch.nn as nn

import torchvision.transforms as transforms

import lpips

from pytorch_msssim import ssim
from torchmetrics.image.fid import FrechetInceptionDistance



class Metricator:
    """
    Computes image quality metrics (PSNR, LPIPS, SSIM) for generated samples against ground truth.
    """
    def __init__(self, device: str = "cuda"):
        self.loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)
        self.device = device
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.met_smpl_gen = torch.Generator()
        self.met_smpl_gen.manual_seed(42)
        self.fid_t = transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR)
        
    @torch.no_grad()
    def measure_metrics(
        self,
        generated_samples: torch.Tensor,
        gt_data: dict,
        N_noisy: int,
        N_clean: int,
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Args:
            output_all_samples (torch.Tensor): All generated samples. Should contain
                N_clean renders from conditioning cameras, N_noisy from denoised views, and
                gt_data["test_imgs"].shape[1] renders from testing viewpoints.
            gt_data (dict): Ground truth images from testing viewpoints.
            N_noisy (int): Number of noisy images in viewset.
            N_clean (int): Number of clean conditioning images in viewset.
        Returns:
            Tuple of lists: (psnrs, lpipses, ssims)
        """
        # assert N_noisy > 0, "N_noisy should be greater than 0"
        num_gt_frames = gt_data["test_imgs"].shape[1]
        expected_frames = N_clean + N_noisy + num_gt_frames
        assert generated_samples.shape[1] == expected_frames, "Wrong number of frames"

        N_test_start = N_clean + N_noisy

        # Resize if needed
        out_img_size = generated_samples.shape[3]
        gt_img_size = gt_data["test_imgs"].shape[3]
        if out_img_size != gt_img_size:
            print(f"Resizing output images from {out_img_size} to {gt_img_size}")
            resizing = transforms.Resize(gt_img_size, interpolation=transforms.InterpolationMode.BILINEAR)
        else:
            resizing = nn.Identity()

        output_test_frames = generated_samples[:, N_test_start:, ...]
        l2_criterion = nn.MSELoss(reduction="none")

        psnrs, lpipses, ssims = [], [], []
        batch_size = generated_samples.shape[0]
        for idx in range(batch_size):
            out_imgs = resizing(output_test_frames[idx])
            gt_imgs = gt_data["test_imgs"][idx]
            fid_idxs = torch.randperm(out_imgs.shape[0], generator=self.met_smpl_gen)[:15]
            self.fid.update(self.fid_t(out_imgs[fid_idxs]), real=False)
            self.fid.update(self.fid_t(gt_imgs[fid_idxs]), real=True)

            # PSNR
            l2_loss = l2_criterion(out_imgs, gt_imgs).mean(dim=[1, 2, 3])
            psnr = -10 * torch.log10(l2_loss)
            psnrs.append(psnr.mean().item())

            # LPIPS
            lpips = self.loss_fn_vgg(out_imgs * 2 - 1, gt_imgs * 2 - 1)
            lpipses.append(lpips.mean().cpu().item())

            # SSIM
            ssim_ = ssim(out_imgs, gt_imgs, data_range=1)
            ssims.append(ssim_.cpu().item())

        return psnrs, lpipses, ssims
    
    def compute_fid(self) -> float:
        """Compute FID after all updates."""
        fid_value = self.fid.compute().item()
        return fid_value
