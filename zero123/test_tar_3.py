import os
import random
import torch
import torchvision.transforms as transforms
import webdataset as wds
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from einops import rearrange

def custom_transform(x):
    return rearrange(x * 2. - 1., 'c h w -> h w c')

def custom_collate(batch, transform):
    """
    Custom collate function to handle tuples in batch samples and apply image transformations.
    """
    batch_out = {'png': [], 'npy': []}
    for sample in batch:
        if len(sample) == 2:
            batch_out['png'].append(transform(sample[0]))
            batch_out['npy'].append(torch.tensor(sample[1]))
    batch_out['png'] = torch.stack(batch_out['png'])
    batch_out['npy'] = torch.stack(batch_out['npy'])
    return batch_out

class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        """
        PyTorch Lightning DataModule for the Objaverse dataset.
        """
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        dataset_config = train if train is not None else validation
        if 'image_transforms' in dataset_config:
            image_transforms = [transforms.Resize(dataset_config['image_transforms']['size'])]
        else:
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(), custom_transform])
        self.image_transforms = transforms.Compose(image_transforms)

    def setup(self, stage=None):
        """
        Setup function to create datasets for different stages.
        """
        print(f"Setting up datasets for stage: {stage}")
        if stage == 'fit' or stage is None:
            self.train_dataset = self.create_dataset(validation=False)
            self.val_dataset = self.create_dataset(validation=True)
        if stage == 'test' or stage == 'predict':
            self.test_dataset = self.create_dataset(validation=self.validation)
        print("Setup complete.")

    def create_dataset(self, validation=False):
        """
        Creates the WebDataset for training or validation.
        """
        tar_path = self.root_dir
        print(f"Creating dataset from: {tar_path} (validation: {validation})")
        dataset = (wds.WebDataset(tar_path)
                   .shuffle(1000)
                   .decode("pil")
                   .select(lambda x: "png" in x and "npy" in x)  # Filter to include only valid samples
                   .to_tuple("png", "npy"))
        print("Dataset created.")
        return dataset

    def train_dataloader(self):
        """
        Returns the training dataloader.
        """
        print("Creating train dataloader.")
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=lambda batch: custom_collate(batch, self.image_transforms))
        print("Train dataloader created.")
        return loader

    def val_dataloader(self):
        """
        Returns the validation dataloader.
        """
        print("Creating validation dataloader.")
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=lambda batch: custom_collate(batch, self.image_transforms))
        print("Validation dataloader created.")
        return loader

    def test_dataloader(self):
        """
        Returns the test dataloader.
        """
        print("Creating test dataloader.")
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=lambda batch: custom_collate(batch, self.image_transforms))
        print("Test dataloader created.")
        return loader

# Usage
root_dir = '/export/compvis-nfs/group/datasets/views_release.tar.gz'
batch_size = 4
total_view = 12
train_config = {'image_transforms': {'size': 256}}
validation_config = {'image_transforms': {'size': 256}}

print("Initializing data module.")
data_module = ObjaverseDataModuleFromConfig(root_dir, batch_size, total_view, train=train_config, validation=validation_config)
data_module.setup('fit')

train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Iterate through a few batches to check
num_batches_to_check = 4
print(f"Checking first {num_batches_to_check} batch of training data:")

for i, batch in enumerate(train_loader):
    print(f"Processing batch {i+1}...")
    if i >= num_batches_to_check:
        break
    print(f"Batch {i+1}:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor with shape {value.shape}")
        else:
            print(f"  {key}: {value}")
    print("\n")

print("Done.")
