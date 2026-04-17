import os
from pathlib import Path
from typing import Dict, List, Tuple

from omegaconf import OmegaConf

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from . import set_seed, num_to_groups
from . import SPGenModel
from . import get_data_manager

from diffusers import FlowMatchEulerDiscreteScheduler
import tqdm
import math
import gc

def model_predictions(
    reconstructor, x, t,
    x_self_cond=None,
    return_all_renders=False,
):

    if return_all_renders:
        model_output = reconstructor(x, t, x_self_cond, eval_return_all_renders=True)
    else:
        model_output = reconstructor(x, t, x_self_cond)
    model_output = model_output['r_imgs'][-1]
    model_output = torch.clamp(model_output, min=-1.0, max=1.0)[:, :, :3, ...]
    if return_all_renders:
        return model_output

    x_start = model_output[:, x_self_cond["x_cond"].shape[1]:, :3, ...]
    x_start = torch.clamp(x_start, min=-1.0, max=1.0)
    return x_start


@torch.no_grad()
def gen_sample(cfg, reconstructor, noise_scheduler: FlowMatchEulerDiscreteScheduler,
               shape, cond, device,
               num_inference_steps=250):

    batch_size = shape[0]

    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = noise_scheduler.timesteps
    

    img = torch.randn(shape, device=device)
    x_start = None

    if cfg.model.unet.self_condition:
        cond["x_self_condition"] = torch.zeros_like(img)

    for i, t in tqdm.tqdm(list(enumerate(timesteps)), desc='sampling loop time step'):
        time_cond = torch.full((batch_size,), t, device=device, dtype=torch.long)
        if i == len(timesteps) - 1:
            assert "test_Rs" in cond.keys(), "No test_Rs and test_Ts provided for final evaluation renders"
            all_target_Ts_loop = cond["test_Ts"]
            all_target_Rs_loop = cond["test_Rs"]
            N_vis = cond["test_Rs"].shape[1]

            cond_vis = {k: v for k, v in cond.items()}
            cond_reference = cond

            # each render in the loop is done separately to manage memory usage
            cond_vis["target_Rs"] = torch.cat([cond_reference["target_Rs"],
                                                all_target_Rs_loop[:, :]], dim=1)
            cond_vis["target_Ts"] = torch.cat([cond_reference["target_Ts"],
                                                all_target_Ts_loop[:, :]], dim=1)
            cond_vis["background"] = torch.cat([cond_reference["background"],
                                            cond["background"][:, :1, ...].expand(batch_size, N_vis, *shape[3:], 3)],
                                            dim=1)
            cond_vis["target_pose_embeds"] = torch.cat([cond_reference["pose_embed"], cond["test_pose_embeds"]], dim=1)
            x_start = []
            for b_idx in range(batch_size):
                x_start_b = model_predictions(
                    reconstructor,
                    img[b_idx:b_idx+1, ...],
                    time_cond[b_idx:b_idx+1, ...],
                    {k: v[b_idx:b_idx+1, ...] for k, v in cond_vis.items()},
                    return_all_renders=True,
                )
                x_start.append(x_start_b)
            x_start = torch.cat(x_start, dim=0)
        else:
            x_start = model_predictions(reconstructor, img, time_cond, cond)

            if cfg.model.unet.self_condition:
                # x_start was last calculated with the correct amount of classifier
                # free guidance - this is the one that will be used for conditioning
                cond["x_self_condition"] = x_start

        if i == len(timesteps) - 1:
            img = x_start
            continue

        predicted_velocity = (img - x_start) / noise_scheduler.sigmas[i].reshape(-1, *((1,) * (img.ndim - 1)))
        img = noise_scheduler.step(predicted_velocity, t, img).prev_sample

    ret = img
    ret = (ret + 1) * 0.5
        
    ret = torch.clamp(ret, 0.0, 1.0)
    return ret


class Generator():
    """ 
    Universal generator for qualitative evaluation
    takes in parameters
      - indexes of images in the dataset to use for camera pose / conditioning
      - number of clean conditioning images
      - number of noisy images in viewset
    loads appropriate model and dataset
    load:
      - volume model
      - diffusion model
      - dataset"""

    def __init__(self, model_path, device, seed=0,
                 num_inference_steps=250,
                 **kwargs):
        # for reproducibility
        set_seed(seed=seed)
        # experiment path should be a folder that contains /hydra folder
        # with the config.yaml file and a model.pth file
        experiment_path = os.path.dirname(model_path)
        cfg = OmegaConf.load(os.path.join(
            experiment_path, ".hydra", "config.yaml"))
        self.experiment_path = experiment_path
        self.cfg = cfg
        self.device = device
        self.num_inference_steps = num_inference_steps

        if cfg['model'].get('type', "volume") != 'query':
            raise ValueError(
                f"Only model.type='query' is supported, got {cfg['model'].get('type')}"
            )
        self.reconstructor = SPGenModel(cfg)

        checkpoint = torch.load(model_path,
                                map_location=self.device)

        rec_params = {}
        for param_name in checkpoint['diffuser']:
            if param_name[:14] == 'reconstructor.':
                rec_params[param_name[14:]] = checkpoint['diffuser'][param_name]
        self.reconstructor.load_state_dict(rec_params)
        self.reconstructor.to(self.device)
        
        self.dataset = None
        self.convert_to_double_conditioning = None
        self.convert_to_single_conditioning = None
        self.n_cond_imgs = None

        if cfg.model.diffuser.type != "RF":
            raise ValueError(
                f"Only model.diffuser.type='RF' is supported in evaluation, got {cfg.model.diffuser.type}"
            )
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=cfg.model.diffuser.steps,
            shift=1.0,
            use_dynamic_shifting=False,
        )
    
    def prepare_dataset(self, split):
        """
        Updates dataset with the appropriate conversion (single vs double conditioning)
        """
        # load dataset - use validation by default
        self.dataset = get_data_manager(self.cfg, split=split,
                                        convert_to_double_conditioning=self.convert_to_double_conditioning,
                                        convert_to_single_conditioning=self.convert_to_single_conditioning,
                                        n_cond_imgs=self.n_cond_imgs)

    def update_dataset(self, N_clean, split):
        if N_clean == 1:
            if not self.convert_to_single_conditioning:
                self.convert_to_single_conditioning = True
                self.convert_to_double_conditioning = False
            self.n_cond_imgs = None
        elif N_clean == 2:
            if not self.convert_to_double_conditioning:
                self.convert_to_single_conditioning = False
                self.convert_to_double_conditioning = True
            self.n_cond_imgs = None
        elif N_clean >= 3:
            # General N-conditioning: pass n_cond_imgs to data managers
            self.convert_to_single_conditioning = False
            self.convert_to_double_conditioning = False
            self.n_cond_imgs = N_clean
        else:
            if not self.convert_to_single_conditioning:
                self.convert_to_single_conditioning = True
                self.convert_to_double_conditioning = False
            self.n_cond_imgs = None
            assert N_clean == 0, "N_clean must be >= 0"

        self.prepare_dataset(split=split)

    @torch.no_grad()
    def generate_samples(self, dataset_idxs, N_clean, N_noisy, split='val'):
        """
        Args:
            dataset_idxs: list of indexes of images in the dataset to use for 
                camera pose / conditioning. 
            N_clean: number of clean conditioning images
            N_noisy: number of noisy images in viewset
        """
        # comment the line below to avoid repeated dataset loading in quantitative eval
        self.update_dataset(N_clean, split)
        gt_data = {"validation_imgs": [],
                   "x_cond": [],
                   "test_imgs": [],
                   "test_occs": []}
        batch_data = {k: [] for k in ["training_imgs",
                                      "validation_imgs",
                                      "x_in",
                                      "x_cond",
                                      "pose_embed",
                                      "val_pose_embeds",
                                      "target_Rs",
                                      "target_Ts",
                                      "background",
                                      "test_pose_embeds",
                                      "test_Rs",
                                      "test_Ts",
                                      "test_imgs",
                                      "test_occs"]}
        if self.cfg.data.get("use_occ", False):
            batch_data["training_occs"] = []
            batch_data["validation_occs"] = []
        for ex_idx in dataset_idxs:
            ex_with_virtual_views = self.dataset.get_item_for_testing(ex_idx, N_noisy)
            for k, v in ex_with_virtual_views.items():
                batch_data[k].append(v.unsqueeze(0).to(self.device))
        for k, v in batch_data.items():
            batch_data[k] = torch.cat(v, dim=0)
        for k in gt_data.keys():
            if k in batch_data.keys():
                gt_data[k] = batch_data[k]

        output_all_samples = gen_sample(
            cfg=self.cfg,
            reconstructor=self.reconstructor,
            noise_scheduler=self.noise_scheduler,
            shape=(len(dataset_idxs), N_noisy, 3,
                   self.cfg.data.input_size[0], self.cfg.data.input_size[1]),
            cond=batch_data,
            device=self.device,
            num_inference_steps=self.num_inference_steps,
        )

        for k in gt_data.keys():
            if k not in batch_data.keys():
                gt_data.pop(k)

        return output_all_samples, gt_data


class Evaluator(nn.Module):
    """
    Universal evaluator for qualitative and quantitative evaluation.
    
    Takes care of:
    - Model loading and initialization
    - Dataset preparation and batching
    - Sample generation with proper conditioning
    - Metric computation and result saving
    - Image saving and visualization
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        seed: int = 0,
        num_inference_steps: int = 250,
        **kwargs
    ):
        super().__init__()
        
        # Initialize generator for model loading and sample generation
        self.generator = Generator(
            model_path=model_path,
            device=device,
            seed=seed,
            num_inference_steps=num_inference_steps,
            **kwargs
        )
        
        # Store configuration
        self.cfg = self.generator.cfg
        self.device = device
        self.seed = seed
        
        # Initialize metricator for evaluation
        from evaluation.metricator import Metricator
        self.metricator = Metricator()
        
        # Evaluation state
        self.current_split = None
        self.current_n_clean = None
        self.current_n_noisy = None

    def _prepare_evaluation_setup(
        self,
        n_clean: int,
        n_noisy: int,
        split: str,
    ) -> None:
        """Prepare dataset and evaluation parameters."""
        # Update dataset configuration if needed
        if (self.current_split != split or 
            self.current_n_clean != n_clean or 
            self.current_n_noisy != n_noisy):
            
            self.generator.update_dataset(
                N_clean=n_clean,
                split=split,
            )
            
            self.current_split = split
            self.current_n_clean = n_clean
            self.current_n_noisy = n_noisy

    def _create_batches(self, dataset_length: int) -> Tuple[List[int], int, int]:
        """Create evaluation batches for a single-device evaluation run."""
        from utils import num_to_groups

        batch_size_per_sample = 32 // (self.current_n_clean + self.current_n_noisy)
        batches = num_to_groups(dataset_length, batch_size_per_sample)

        return batches, 0, dataset_length

    def _save_sample_images(
        self,
        generated_samples: torch.Tensor,
        gt_data: Dict[str, torch.Tensor],
        example_ids: List[str],
        batch_start: int,
        save_dir: str,
    ) -> None:
        """Save generated and ground truth images for visualization."""
        n_test_start = self.current_n_clean + self.current_n_noisy
        
        for local_idx, sample in enumerate(generated_samples):
            global_idx = batch_start + local_idx
            example_id = example_ids[global_idx]
            
            sample_save_dir = Path(save_dir) / example_id
            sample_save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save test views (novel views)
            test_samples = sample[n_test_start:]
            test_gt = gt_data["test_imgs"][local_idx]
        
            for rot_idx in range(len(test_samples)):
                img_gen = test_samples[rot_idx]
                img_gt = test_gt[rot_idx]
                
                # Concatenate generated and GT images side by side
                output_frame = torch.cat([img_gen, img_gt], dim=2)
                output_frame = output_frame.permute(1, 2, 0).cpu().numpy()
                output_frame = (output_frame * 255).astype('uint8')
                
                output_image = Image.fromarray(output_frame)
                output_image.save(sample_save_dir / f"frame_{rot_idx:03d}.png")

    @torch.no_grad()
    def evaluate_samples(
        self,
        n_clean: int,
        n_noisy: int,
        split: str = "val",
        save_output: bool = False,
        output_dir: str = None,
    ) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[str], float]:
        """
        Main evaluation function that generates samples and computes metrics.
        
        Args:
            n_clean: Number of clean conditioning images
            n_noisy: Number of noisy images in viewset
            split: Dataset split to evaluate on
            save_output: Whether to save generated images
            output_dir: Directory to save outputs
            
        Returns:
            Tuple of (psnrs, lpipses, ssims, example_ids)
        """
        # Prepare evaluation setup
        self._prepare_evaluation_setup(
            n_clean=n_clean,
            n_noisy=n_noisy,
            split=split,
        )
        
        dataset_length = len(self.generator.dataset)
        batches, chunk_start, chunk_end = self._create_batches(dataset_length)
        
        # Initialize result storage
        all_psnrs = []
        all_lpipses = []
        all_ssims = []
        example_ids = []
        
        batch_start = chunk_start
        
        for cur_batch_size in batches:
            # Prepare batch indices and example IDs
            ex_idxs = list(range(batch_start, batch_start + cur_batch_size))
            
            for ex_idx in ex_idxs:
                example_id = self.generator.dataset.get_example_id(ex_idx)
                example_ids.append(example_id)
                
                # Initialize metric lists for this example
                for metric_list in [all_psnrs, all_lpipses, all_ssims]:
                    metric_list.append([])
            
            generated_samples, gt_data = self.generator.generate_samples(
                dataset_idxs=ex_idxs,
                N_clean=n_clean,
                N_noisy=n_noisy,
                split=split,
            )
            
            psnrs, lpipses, ssims = self.metricator.measure_metrics(
                generated_samples=generated_samples,
                gt_data=gt_data,
                N_clean=n_clean,
                N_noisy=n_noisy,
            )
            
            for local_idx, global_idx in enumerate(ex_idxs):
                result_idx = global_idx - chunk_start
                all_psnrs[result_idx].append(psnrs[local_idx])
                all_lpipses[result_idx].append(lpipses[local_idx])
                all_ssims[result_idx].append(ssims[local_idx])
            
            if save_output and output_dir:
                self._save_sample_images(
                    generated_samples=generated_samples,
                    gt_data=gt_data,
                    example_ids=example_ids,
                    batch_start=batch_start,
                    save_dir=output_dir,
                )
            
            batch_start += cur_batch_size
            print(f"Evaluated {len(all_psnrs)} out of {dataset_length} examples")
            gc.collect()
            torch.cuda.empty_cache()
        
        fid = self.metricator.compute_fid()

        return all_psnrs, all_lpipses, all_ssims, example_ids, fid

    @staticmethod
    def compute_summary_metrics(
        all_psnrs: List[List[float]],
        all_lpipses: List[List[float]],
        all_ssims: List[List[float]],
    ) -> Dict[str, float]:
        """Compute summary statistics from evaluation results."""
        # Filter out NaN values
        flat_psnrs = [score for scores in all_psnrs for score in scores if math.isfinite(score)]
        flat_lpipses = [score for scores in all_lpipses for score in scores if math.isfinite(score)]
        flat_ssims = [score for scores in all_ssims for score in scores if math.isfinite(score)]
        
        summary = {
            "mean_psnr": np.mean(flat_psnrs),
            "mean_lpips": np.mean(flat_lpipses),
            "mean_ssim": np.mean(flat_ssims),
            "num_examples": len(all_psnrs),
            "total_samples": len(flat_psnrs),
        }
        
        return summary
