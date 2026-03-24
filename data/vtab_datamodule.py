from typing import Optional, Union

import lightning.pytorch as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms


def add_index_column(ds, dataset_name, split_name):
    ds = ds.add_column(
        "index",
        [f"{dataset_name}/{split_name}_{i:08d}" for i in range(len(ds))],
    )
    return ds


class ProcessDataTransform:
    """Picklable transform for HuggingFace set_transform.

    Defined at module level (not as a closure) so that it can be pickled
    by spawn-based multiprocessing on macOS / Python 3.12+.
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, examples):
        out_batch = {}
        out_batch["x"] = torch.stack(
            [self.transform(img.convert("RGB")) for img in examples["image"]]
        )
        out_batch["y"] = torch.tensor(examples["label"])
        out_batch["index"] = examples["index"]
        return out_batch


class VTABDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name="bramtoula/vtab_cifar100",
        batch_size=32,
        eval_batch_size=256,
        shuffle_val_loader: bool = False,
        num_workers: int = 8,
        train_split_name="train800",
        val_split_name="val200",
        test_split_name="test",
        force_resize: Optional[Union[int, tuple]] = (224, 224),
        interpolation="bicubic",
        limit_train_samples: Optional[
            int
        ] = None,  # New parameter for limiting training samples
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        # Create a simple transform that only resizes images and converts to tensor
        if isinstance(self.hparams.force_resize, int):
            size = (self.hparams.force_resize, self.hparams.force_resize)
        else:
            size = self.hparams.force_resize

        if self.hparams.interpolation == "bicubic":
            interpolation = transforms.InterpolationMode.BICUBIC
        elif self.hparams.interpolation == "bilinear":
            interpolation = transforms.InterpolationMode.BILINEAR
        else:
            raise ValueError(
                f"Unsupported interpolation mode: {self.hparams.interpolation}"
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize(size, interpolation=interpolation),
                transforms.ToTensor(),
            ]
        )

        # Load the dataset splits
        self.ds_train = load_dataset(
            self.hparams.dataset_name, split=self.hparams.train_split_name
        )
        self.ds_valid = load_dataset(
            self.hparams.dataset_name, split=self.hparams.val_split_name
        )
        self.ds_test = load_dataset(
            self.hparams.dataset_name, split=self.hparams.test_split_name
        )

        # Limit training samples if specified
        if (
            self.hparams.limit_train_samples is not None
            and self.hparams.limit_train_samples < len(self.ds_train)
        ):
            self.ds_train = self.ds_train.shuffle().select(
                range(self.hparams.limit_train_samples)
            )

        self.ds_train = add_index_column(
            self.ds_train, self.hparams.dataset_name, self.hparams.train_split_name
        )
        self.ds_valid = add_index_column(
            self.ds_valid, self.hparams.dataset_name, self.hparams.val_split_name
        )
        self.ds_test = add_index_column(
            self.ds_test, self.hparams.dataset_name, self.hparams.test_split_name
        )

    def setup(self, stage=None):
        process_fn = ProcessDataTransform(self.transform)
        self.ds_train.set_transform(process_fn)
        self.ds_valid.set_transform(process_fn)
        self.ds_test.set_transform(process_fn)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_valid,
            batch_size=self.hparams.eval_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=self.hparams.shuffle_val_loader,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.hparams.eval_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            persistent_workers=self.hparams.num_workers > 0,
        )
