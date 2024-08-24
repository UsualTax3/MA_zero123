import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from einops import rearrange
import pytorch_lightning as pl
import webdataset as wds

def custom_transform(x):
    return rearrange(x * 2. - 1., 'c h w -> h w c')

def custom_collate(batch, transform, total_view):
    batch_out = {'image_target': [], 'image_cond': [], 'T': []}

    for sample in batch:
        pngs = [sample[key] for key in sample.keys() if 'png' in key]
        npys = [sample[key] for key in sample.keys() if 'npy' in key]

        index_target, index_cond = random.sample(range(total_view), 2)

        target_im = transform(pngs[index_target])
        cond_im = transform(pngs[index_cond])

        target_RT = npys[index_target]
        cond_RT = npys[index_cond]

        batch_out['image_target'].append(target_im)
        batch_out['image_cond'].append(cond_im)
        batch_out['T'].append(get_T(target_RT, cond_RT))

    batch_out['image_target'] = torch.stack(batch_out['image_target'])
    batch_out['image_cond'] = torch.stack(batch_out['image_cond'])
    batch_out['T'] = torch.stack(batch_out['T'])

    return batch_out

def cartesian_to_spherical(xyz):
    xy = xyz[0]**2 + xyz[1]**2
    z = np.sqrt(xy + xyz[2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[2])
    azimuth = np.arctan2(xyz[1], xyz[0])
    return np.array([theta, azimuth, z])

def get_T(target_RT, cond_RT):
    R, T = target_RT[:3, :3], target_RT[:, -1]
    T_target = -R.T @ T

    R, T = cond_RT[:3, :3], cond_RT[:, -1]
    T_cond = -R.T @ T

    theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(T_cond)
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target)

    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * np.pi)
    d_z = z_target - z_cond

    d_T = torch.tensor([d_theta.item(), np.sin(d_azimuth.item()), np.cos(d_azimuth.item()), d_z.item()])
    return d_T

class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        dataset_config = train if train is not None else validation if validation is not None else {}
        if 'image_transforms' in dataset_config:
            image_transforms = [transforms.Resize(dataset_config['image_transforms']['size'])]
        else:
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(), custom_transform])
        self.image_transforms = transforms.Compose(image_transforms)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self.create_dataset(validation=False)
            self.val_dataset = self.create_dataset(validation=True)
        if stage == 'test' or stage == 'predict':
            self.test_dataset = self.create_dataset(validation=self.validation)

    def inspect_sample_keys(self):
        tar_path = self.root_dir
        sample_dataset = wds.WebDataset(tar_path).shuffle(1000).decode("pil")
        for sample in sample_dataset:
            print(f"Sample keys: {sample.keys()}")
            for key, value in sample.items():
                if key in ['__key__', '__url__']:
                    print(f"Key: {key}\n  Metadata: {value}")
                else:
                    print(f"Key: {key}\n  Data Type: {type(value)}")
            break

    def create_dataset(self, validation=False):
        tar_path = self.root_dir
        
        self.inspect_sample_keys()  # Inspect keys before creating the dataset

        # Create a dictionary to store paired samples
        paired_samples = {}
        print("Starting to pair samples...")

        # Iterate over the dataset and pair samples based on the keys
        sample_dataset = wds.WebDataset(tar_path).shuffle(1000).decode("pil")
        for sample in sample_dataset:
            key = sample['__key__']
            #print(f"Processing sample with key: {key}")
            if key not in paired_samples:
                paired_samples[key] = {}
            paired_samples[key].update(sample)
            #print(f"Pairing sample with key: {key}")

        print("Finished pairing samples.")
        
        # Filter out incomplete pairs
        filtered_samples = [v for k, v in paired_samples.items() if 'png' in v and 'npy' in v]
        print(f"Number of paired samples: {len(filtered_samples)}")

        # Debug: Print a sample of filtered_samples
        print("Sample of filtered samples:", filtered_samples[0] if filtered_samples else "No samples found")

        # Create a new WebDataset object from the filtered samples
        dataset = wds.WebDataset(tar_path).shuffle(1000).decode("pil").to_tuple("png", "npy")
        
        return dataset

    def train_dataloader(self):
        print("Creating train DataLoader...")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=lambda batch: custom_collate(batch, self.image_transforms, self.total_view))

    def val_dataloader(self):
        print("Creating validation DataLoader...")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=lambda batch: custom_collate(batch, self.image_transforms, self.total_view))

    def test_dataloader(self):
        print("Creating test DataLoader...")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=lambda batch: custom_collate(batch, self.image_transforms, self.total_view))

# Instantiate the data module and inspect the keys
data_module = ObjaverseDataModuleFromConfig(
    root_dir='/export/compvis-nfs/group/datasets/views_release.tar.gz', 
    batch_size=4, 
    total_view=12,
    train={'image_transforms': {'size': (256, 256)}}
)
data_module.inspect_sample_keys()

data_module.setup('fit')
train_loader = data_module.train_dataloader()

for batch in train_loader:
    print("Processing a new batch...")
    if batch is not None:
        print(f"Batch image_target shape: {batch['image_target'].shape}")
        print(f"Batch image_cond shape: {batch['image_cond'].shape}")
        print(f"Batch T shape: {batch['T'].shape}")
    else:
        print("Skipped a batch due to missing data.")
    break
