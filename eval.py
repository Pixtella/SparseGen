import hydra
import os
import torch
from omegaconf import DictConfig, OmegaConf
from evaluation.generator import Evaluator
from utils import set_seed


@hydra.main(version_base=None, config_path="configs_eval", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main evaluation function using the new Evaluator class."""
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.get("seed", 0))

    device = torch.device("cuda")
    
    # Initialize evaluator
    evaluator = Evaluator(
        model_path=cfg.model_path,
        device=device,
        seed=cfg.get("seed", 0),
        num_inference_steps=cfg.get("num_inference_steps", 250),
    )
    
    # Run evaluation
    all_psnrs, all_lpipses, all_ssims, example_ids, fid = evaluator.evaluate_samples(
        n_clean=cfg.N_clean,
        n_noisy=cfg.N_noisy,
        split=cfg.eval.split,
        save_output=cfg.get("save_output", False),
        output_dir=os.getcwd() if cfg.get("save_output", False) else None,
    )
    
    # Compute and print summary metrics
    summary_metrics = evaluator.compute_summary_metrics(all_psnrs, all_lpipses, all_ssims)
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Number of examples: {summary_metrics['num_examples']}")
    print(f"Total samples: {summary_metrics['total_samples']}")
    print(f"PSNR: {summary_metrics['mean_psnr']:.5f}")
    print(f"LPIPS: {summary_metrics['mean_lpips']:.5f}")
    print(f"SSIM: {summary_metrics['mean_ssim']:.5f}")
    print(f"FID: {fid:.5f}")
    print("="*50)


if __name__ == "__main__":
    main()
