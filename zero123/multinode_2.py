import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from einops import rearrange
import pytorch_lightning as pl
import webdataset as wds
from collections import defaultdict

def custom_transform(x):
    return rearrange(x * 2. - 1., 'c h w -> h w c')

def custom_collate(batch, transform, total_view):
    batch_out = {'image_target': [], 'image_cond': [], 'T': []}

    for sample in batch:
        print(f"Processing sample in collate: {sample}")

        pngs = sample[0]
        npys = sample[1]

        print(f"pngs: {pngs}")
        print(f"npys: {npys}")

        target_im = transform(pngs)
        cond_im = transform(pngs)

        target_RT = npys
        cond_RT = npys

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
                 test=None, num_workers=4, multinode=False, **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view
        self.multinode = multinode

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

    def filter_fn(self, sample):
        return 'png' in sample and 'npy' in sample

    def pair_samples(self, sample):
        key = sample['__key__']
        base_key = '/'.join(key.split('/')[:-1])
        sample['__main__'] = base_key
        if base_key not in self.paired_samples:
            self.paired_samples[base_key] = {}
        self.paired_samples[base_key].update(sample)
        if 'png' in self.paired_samples[base_key] and 'npy' in self.paired_samples[base_key]:
            print(f"Paired sample with key: {key}")
            return self.paired_samples.pop(base_key)
        return None

    def group_by_main_key(self):
        grouped_samples = defaultdict(lambda: {'png': [], 'npy': []})

        for key, sample in self.paired_samples.items():
            main_key = sample['__main__']
            if 'png' in sample:
                grouped_samples[main_key]['png'].append(sample['png'])
            if 'npy' in sample:
                grouped_samples[main_key]['npy'].append(sample['npy'])

        # Filter out incomplete groups
        complete_samples = {k: v for k, v in grouped_samples.items() if v['png'] and v['npy']}
        return complete_samples

    def create_dataset(self, validation=False):
        tar_path = self.root_dir
        nodesplitter = wds.shardlists.split_by_node if self.multinode else wds.shardlists.single_node_only

        self.paired_samples = {}
        
        dataset = (wds.WebDataset(tar_path, nodesplitter=nodesplitter)
                   .shuffle(100)
                   .decode("pil")
                   .map(self.pair_samples)
                   .select(lambda x: x is not None)
                   .map(self.group_by_main_key)
                   .to_tuple("png", "npy"))

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
    train={'image_transforms': {'size': (256, 256)}},
    multinode=True
)

data_module.setup('fit')

# Print some grouped samples
print("Inspecting grouped samples:")
grouped_samples = data_module.group_by_main_key()
for main_key, samples in list(grouped_samples.items())[:5]:  # Print the first 5 grouped samples
    print(f"Main key: {main_key}")
    print(f"PNGs: {samples['png']}")
    print(f"NPYs: {samples['npy']}")
    print("------")

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
