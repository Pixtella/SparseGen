from . import SRNDataset

def get_data_manager(cfg, split='train',
                     convert_to_double_conditioning=None,
                     convert_to_single_conditioning=None,
                     n_cond_imgs=None,
                     **kwargs):

    assert split in ['train', 'val', 'test'], "Invalid split"

    if convert_to_double_conditioning is None:
        if cfg.data.two_training_imgs_per_example:
            convert_to_double_conditioning=True
        else:
            convert_to_double_conditioning=False

    if convert_to_single_conditioning is None:
        if cfg.data.one_training_img_per_example:
            convert_to_single_conditioning=True
        else:
            convert_to_single_conditioning=False

    if cfg.data.dataset_type == "srn":
        dataset = SRNDataset(cfg,
                            convert_to_single_conditioning,
                            convert_to_double_conditioning,
                            dataset_name=split,
                            n_cond_imgs=n_cond_imgs,
                            **kwargs)
    else:
        raise NotImplementedError(
            f"Dataset type {cfg.data.dataset_type} not implemented. Only SRN is supported."
        )
        
    return dataset
