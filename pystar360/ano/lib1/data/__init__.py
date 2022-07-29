

from typing import Union

from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningDataModule

from .folder import FolderDataModule
from .inference import InferenceDataset


def get_datamodule(config: Union[DictConfig, ListConfig]) -> LightningDataModule:
    """Get  Datamodule.

    Args:
        config (Union[DictConfig, ListConfig]): Configuration of the  model.

    Returns:
        PyTorch Lightning DataModule
    """
    datamodule: LightningDataModule

    if config.dataset.format.lower() == "folder":
        datamodule = FolderDataModule(
            root=config.dataset.path,
            normal_dir=config.dataset.normal_dir,
            abnormal_dir=config.dataset.abnormal_dir,
            task=config.dataset.task,
            normal_test_dir=config.dataset.normal_test_dir,
            mask_dir=config.dataset.mask,
            extensions=config.dataset.extensions,
            split_ratio=config.dataset.split_ratio,
            seed=config.dataset.seed,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            train_batch_size=config.dataset.train_batch_size,
            test_batch_size=config.dataset.test_batch_size,
            num_workers=config.dataset.num_workers,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_val=config.dataset.transform_config.val,
            create_validation_set=config.dataset.create_validation_set,
        )
    else:
        raise ValueError(
            "Unknown dataset! \n"
        )

    return datamodule


__all__ = [
    "get_datamodule",
    # "BTechDataModule",
    "FolderDataModule",
    "InferenceDataset",
    # "MVTecDataModule",
]
